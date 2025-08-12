import awkward
import numpy
import warnings
from functools import partial, reduce
import operator


def getDataJecTag(jec_pars, dataset):
    """
    helper function that returns the correct JEC tag for the specific run
    """
    jec_data_tag_dict = jec_pars["jec_data_tags"]
    print(f"jec_data_tag_dict: {jec_data_tag_dict}")
    for jec_tag, runs in jec_data_tag_dict.items():
        for run in runs:
            print(f"run: {run}")
            if run in dataset:
                return jec_tag

    # if nothing gets returned, we have an issue
    print("ERROR: No JEC data TAG was GIVEN")
    raise ValueError

def ApplyJetCorrections(jec_pars, year, dataset):
    """
    taken from Andrea
    """
    is_mc = not "data" in dataset
    print(f"dataset: {dataset}")
    print(f"is_mc: {is_mc}")
    print(f"jec_pars: {jec_pars}")
    jet_algo = jec_pars["jet_algorithm"]
    jec_tag =  jec_parameters["jec_tags"][year] if is_mc else getDataJecTag(jec_pars, dataset)
    jec_levels = jec_pars["jec_levels_mc"] if is_mc else jec_pars["jec_levels_data"]
    if is_mc:
        jer_tag = jec_pars["jer_tags"] 
        junc_types = jec_pars["jec_unc_to_consider"]
        junc_types = ["Regrouped_" + junc_type for junc_type in junc_types] # add "Regrouped" for each variation
    else: 
        jer_tag = None
        junc_types = None
    print(f"jet_algo: {jet_algo}")
    print(f"jec_tag: {jec_tag}")
    print(f"jec_levels: {jec_levels}")
    print(f"jer_tag: {jer_tag}")
    print(f"junc_types: {junc_types}")

    json_path = f"/work/users/yun79/valerie/fork/copperheadV2/data/POG/JME/{year}_UL/jet_jerc.json.gz" # Hard code for now
    # Create JECStack for clib scenario
    jec_stack = JECStack(
        jec_tag=jec_tag,
        jec_levels=jec_levels,
        jer_tag=jer_tag,
        jet_algo=jet_algo,
        junc_types=junc_types,
        json_path=json_path,
        use_clib=True,
        savecorr=False
    )

    # Name map for jet or MET corrections
    name_map = {
        'JetPt': 'pt',
        'JetMass': 'mass',
        'JetEta': 'eta',
        'JetPhi': 'phi',
        'JetA': 'area',
        'ptGenJet': 'pt_gen',
        'ptRaw': 'pt_raw',
        'massRaw': 'mass_raw',
        'Rho': 'rho',
    }
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    return jet_factory






_stack_parts = ["jec", "junc", "jer", "jersf"]
_MIN_JET_ENERGY = numpy.array(1e-2, dtype=numpy.float32)
_ONE_F32 = numpy.array(1.0, dtype=numpy.float32)
_ZERO_F32 = numpy.array(0.0, dtype=numpy.float32)
_JERSF_FORM = {
    "class": "NumpyArray",
    "inner_shape": [3],
    "itemsize": 4,
    "format": "f",
    "primitive": "float32",
}

def rewrap_recordarray(layout, depth, data):
    if isinstance(layout, awkward.layout.RecordArray):
        return lambda: data
    return None

def awkward_rewrap(arr, like_what, gfunc):
    behavior = awkward._util.behaviorof(like_what)
    func = partial(gfunc, data=arr.layout)
    layout = awkward.operations.convert.to_layout(like_what)
    newlayout = awkward._util.recursively_apply(layout, func)
    return awkward._util.wrap(newlayout, behavior=behavior)

def rand_gauss(item, randomstate):
    def getfunction(layout, depth):
        if isinstance(layout, awkward.layout.NumpyArray) or not isinstance(
            layout, (awkward.layout.Content, awkward.partition.PartitionedArray)
        ):
            return lambda: awkward.layout.NumpyArray(
                randomstate.normal(size=len(layout)).astype(numpy.float32)
            )
        return None
    out = awkward._util.recursively_apply(
        awkward.operations.convert.to_layout(item), getfunction
    )
    assert out is not None
    return awkward._util.wrap(out, awkward._util.behaviorof(item))

def jer_smear(
    variation,
    forceStochastic,
    pt_gen,
    jetPt,
    etaJet,
    jet_energy_resolution,
    jet_resolution_rand_gauss,
    jet_energy_resolution_scale_factor,
):
    pt_gen = pt_gen if not forceStochastic else None
    if not isinstance(jetPt, awkward.highlevel.Array):
        raise Exception("'jetPt' must be an awkward array of some kind!")
    if forceStochastic:
        pt_gen = awkward.without_parameters(awkward.zeros_like(jetPt))

    jersmear = jet_energy_resolution * jet_resolution_rand_gauss
    jersf = jet_energy_resolution_scale_factor[:, variation]
    deltaPtRel = (jetPt - pt_gen) / jetPt
    doHybrid = (pt_gen > 0) & (numpy.abs(deltaPtRel) < 3 * jet_energy_resolution)
    detSmear = 1 + (jersf - 1) * deltaPtRel
    stochSmear = 1 + numpy.sqrt(numpy.maximum(jersf**2 - 1, 0)) * jersmear

    min_jet_pt = _MIN_JET_ENERGY / numpy.cosh(etaJet)
    min_jet_pt_corr = min_jet_pt / jetPt
    smearfact = awkward.where(doHybrid, detSmear, stochSmear)
    smearfact = awkward.where(
        (smearfact * jetPt) < min_jet_pt, min_jet_pt_corr, smearfact
    )

    def getfunction(layout, depth):
        if isinstance(layout, awkward.layout.NumpyArray) or not isinstance(
            layout, (awkward.layout.Content, awkward.partition.PartitionedArray)
        ):
            return lambda: awkward.layout.NumpyArray(smearfact)
        return None

    smearfact = awkward._util.recursively_apply(
        awkward.operations.convert.to_layout(jetPt), getfunction
    )
    smearfact = awkward._util.wrap(smearfact, awkward._util.behaviorof(jetPt))
    return smearfact

# Wrapper function to apply jec corrections
def rawvar_jec(jecval, rawvar, lazy_cache):
    return awkward.virtual(
        operator.mul,
        args=(jecval, rawvar),
        cache=lazy_cache,
    )

def get_corr_inputs(jets, corr_obj, name_map, cache=None, corrections=None):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """

    if corrections is None:
        input_values = [awkward.flatten(jets[name_map[inp.name]]) for inp in corr_obj.inputs if (inp.name != "systematic")]
    else:
        ## This is needed to propagate the previous level of corrections, before applying the next one
        input_values = []
        for inp in corr_obj.inputs:
            if inp.name == "systematic":
                continue
            elif inp.name == "JetPt":
                rawvar = awkward.flatten(jets[name_map[inp.name]])
                init_input_value = partial(rawvar_jec, rawvar=rawvar, lazy_cache=cache)
                input_value = init_input_value(jecval=corrections)
            else:
                input_value = awkward.flatten(jets[name_map[inp.name]])
            input_values.append(input_value)
    return input_values


class CorrectedJetsFactory(object):
    def __init__(self, name_map, jec_stack):
        if not isinstance(jec_stack, JECStack):
            raise TypeError("jec_stack must be an instance of JECStack")

        self.tool = "clib" if jec_stack.use_clib else "jecstack"
        self.forceStochastic = False

        # Handle name map for raw pt and mass
        if "ptRaw" not in name_map or name_map["ptRaw"] is None:
            warnings.warn(
                "There is no name mapping for ptRaw,"
                " CorrectedJets will assume that <object>.pt is raw pt!"
            )
            name_map["ptRaw"] = name_map["JetPt"] + "_raw"
        self.treat_pt_as_raw = "ptRaw" not in name_map

        if "massRaw" not in name_map or name_map["massRaw"] is None:
            warnings.warn(
                "There is no name mapping for massRaw,"
                " CorrectedJets will assume that <object>.mass is raw mass!"
            )
            name_map["massRaw"] = name_map["JetMass"] + "_raw"

        self.jec_stack = jec_stack
        self.name_map = name_map

        if self.jec_stack.use_clib:
            # For clib scenario, load corrections from json_path
            self.load_corrections_clib()
        else:
            # For non-clib scenario, use the provided corrections (e.g., JEC/JER)
            self.load_corrections_jecstack()

        if "ptGenJet" not in name_map:
            warnings.warn(
                'Input JaggedCandidateArray must have "ptGenJet" in order to apply hybrid JER smearing method. Stochastic smearing will be applied.'
            )
            self.forceStochastic = True

    def load_corrections_clib(self):
        """Load the corrections from correctionlib using the json_path in JECStack."""
        self.corrections = self.jec_stack.corrections

    def load_corrections_jecstack(self):
        """Use the corrections provided in the JECStack for non-clib scenario."""
        self.corrections = self.jec_stack.corrections

        # Ensure all required inputs have mappings
        total_signature = set()
        for part in _stack_parts:
            attr = getattr(self.jec_stack, part)
            if attr is not None:
                total_signature.update(attr.signature)

        missing = total_signature - set(self.name_map.keys())
        if len(missing) > 0:
            raise Exception(
                f"Missing mapping of {missing} in name_map!" +
                " Cannot evaluate jet corrections!" +
                " Please supply mappings for these variables!"
            )

    def build(self, jets, lazy_cache):
        if lazy_cache is None:
            raise Exception("CorrectedJetsFactory requires an awkward-array cache to function correctly.")
        lazy_cache = awkward._util.MappingProxy.maybe_wrap(lazy_cache)
        if not isinstance(jets, awkward.highlevel.Array):
            raise Exception("'jets' must be an awkward > 1.0.0 array of some kind!")

        # THESE ARE THE ATTRIBUTES OF THE JET COLLECTION
        fields = awkward.fields(jets)
        if len(fields) == 0:
            raise Exception("Empty record, please pass a jet object with at least {self.real_sig} defined!")

        out = awkward.flatten(jets)
        wrap = partial(awkward_rewrap, like_what=jets, gfunc=rewrap_recordarray)
        scalar_form = awkward.without_parameters(out[self.name_map["ptRaw"]]).layout.form

        in_dict = {field: out[field] for field in fields}
        out_dict = dict(in_dict)

        # Add original values
        out_dict[self.name_map["JetPt"] + "_orig"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_orig"] = out_dict[self.name_map["JetMass"]]
        if self.treat_pt_as_raw:
            out_dict[self.name_map["ptRaw"]] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["massRaw"]] = out_dict[self.name_map["JetMass"]]

        jec_name_map = dict(self.name_map)
        jec_name_map["JetPt"] = jec_name_map["ptRaw"]
        jec_name_map["JetMass"] = jec_name_map["massRaw"]

        # Apply JEC corrections based on scenario
        total_correction = None
        if self.tool == "jecstack":
            if self.jec_stack.jec is not None:
                jec_args = {
                    k: out_dict[jec_name_map[k]] for k in self.jec_stack.jec.signature
                }
                total_correction = self.jec_stack.jec.getCorrection(
                    **jec_args, form=scalar_form, lazy_cache=lazy_cache
                )
            else:
                total_correction = awkward.ones_like(out_dict[self.name_map["JetPt"]])

        elif self.tool == "clib":
            corrections_list = []

            for lvl in self.jec_stack.jec_names_clib:
                cumCorr = None
                if len(corrections_list) > 0:
                    ones = numpy.ones_like(corrections_list[-1], dtype=numpy.float32)
                    cumCorr = reduce(lambda x, y: y * x, corrections_list, ones).astype(dtype=numpy.float32)

                sf = self.corrections.get(lvl, None)
                if sf is None:
                    raise ValueError(f"Correction {lvl} not found in self.corrections")

                ## This automatically apply the previous levels of correction, when needed
                inputs = get_corr_inputs(jets=jets, corr_obj=sf, name_map=jec_name_map, cache=lazy_cache, corrections=cumCorr)
                correction = sf.evaluate(*inputs).astype(dtype=numpy.float32)
                corrections_list.append(correction)
                if total_correction is None:
                    total_correction = numpy.ones_like(correction, dtype=numpy.float32)
                total_correction *= correction

                if self.jec_stack.savecorr:
                    jec_lvl_tag = "_jec_" + lvl

                    out_dict[f"jet_energy_correction_{lvl}"] = correction
                    init_pt_lvl = partial(
                        awkward.virtual,
                        operator.mul,
                        args=(out_dict[f"jet_energy_correction_{lvl}"], out_dict[self.name_map["ptRaw"]]),
                        cache=lazy_cache,
                    )
                    init_mass_lvl = partial(
                        awkward.virtual,
                        operator.mul,
                        args=(out_dict[f"jet_energy_correction_{lvl}"], out_dict[self.name_map["massRaw"]]),
                        cache=lazy_cache,
                    )
                    out_dict[self.name_map["JetPt"] + f"_{lvl}"] = init_pt_lvl(length=len(out), form=scalar_form)
                    out_dict[self.name_map["JetMass"] + f"_{lvl}"] = init_mass_lvl(length=len(out), form=scalar_form)

                    out_dict[self.name_map["JetPt"] + jec_lvl_tag] = out_dict[self.name_map["JetPt"] + f"_{lvl}"]
                    out_dict[self.name_map["JetMass"] + jec_lvl_tag] = out_dict[self.name_map["JetMass"] + f"_{lvl}"]

        out_dict["jet_energy_correction"] = total_correction

        # Finally, the lazy binding to the JEC
        init_pt = partial(
            awkward.virtual,
            operator.mul,
            args=(out_dict["jet_energy_correction"], out_dict[self.name_map["ptRaw"]]),
            cache=lazy_cache,
        )
        init_mass = partial(
            awkward.virtual,
            operator.mul,
            args=(
                out_dict["jet_energy_correction"],
                out_dict[self.name_map["massRaw"]],
            ),
            cache=lazy_cache,
        )

        out_dict[self.name_map["JetPt"]] = init_pt(length=len(out), form=scalar_form)
        out_dict[self.name_map["JetMass"]] = init_mass(length=len(out), form=scalar_form)

        out_dict[self.name_map["JetPt"] + "_jec"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_jec"] = out_dict[self.name_map["JetMass"]]

        has_jer = False
        if self.tool == "jecstack":
            if self.jec_stack.jer is not None and self.jec_stack.jersf is not None:
                has_jer = True
        elif self.tool == "clib":
            has_jer = len(self.jec_stack.jer_names_clib) > 0

        if has_jer:
            jer_name_map = dict(self.name_map)
            jer_name_map["JetPt"] = jer_name_map["JetPt"] + "_jec"
            jer_name_map["JetMass"] = jer_name_map["JetMass"] + "_jec"

            if self.tool == "jecstack":
                jer_args = {
                    k: out_dict[jer_name_map[k]] for k in self.jec_stack.jer.signature
                }
                out_dict["jet_energy_resolution"] = self.jec_stack.jer.getResolution(
                    **jer_args, form=scalar_form, lazy_cache=lazy_cache
                )

                jersf_args = {
                    k: out_dict[jer_name_map[k]] for k in self.jec_stack.jersf.signature
                }
                out_dict["jet_energy_resolution_scale_factor"] = self.jec_stack.jersf.getScaleFactor(
                    **jersf_args, form=_JERSF_FORM, lazy_cache=lazy_cache
                )

            elif self.tool == "clib":
                # Prepare for clib-based corrections
                jer_out_parms = out.layout.parameters
                jer_out_parms["corrected"] = True
                jer_out = awkward.zip(
                    out_dict, depth_limit=1, parameters=jer_out_parms, behavior=out.behavior
                )
                jerjets = wrap(jer_out)

                for jer_entry in self.jec_stack.jer_names_clib:
                    outtag = "jet_energy_resolution"
                    jer_entry = jer_entry.replace("SF", "ScaleFactor")
                    sf = self.corrections[jer_entry]
                    inputs = get_corr_inputs(jets=jerjets, corr_obj=sf, name_map=jer_name_map)
                    if "ScaleFactor" in jer_entry:
                        outtag += "_scale_factor"
                        correction = awkward.Array([
                            sf.evaluate(*inputs, "nom").astype(dtype=numpy.float32),
                            sf.evaluate(*inputs, "up").astype(dtype=numpy.float32),
                            sf.evaluate(*inputs, "down").astype(dtype=numpy.float32),
                        ])
                        correction = awkward.concatenate([
                            correction[0][:, numpy.newaxis],
                            correction[1][:, numpy.newaxis],
                            correction[2][:, numpy.newaxis]
                        ], axis=1)
                    else:
                        correction = awkward.Array(
                            sf.evaluate(*inputs).astype(dtype=numpy.float32),
                        )

                    out_dict[outtag] = correction

                del jerjets

            # Gaussian smearing
            seeds = numpy.array(out_dict[self.name_map["JetPt"] + "_orig"])[[0, -1]].view("i4")
            out_dict["jet_resolution_rand_gauss"] = awkward.virtual(
                rand_gauss,
                args=(
                    out_dict[self.name_map["JetPt"] + "_orig"],
                    numpy.random.Generator(numpy.random.PCG64(seeds)),
                ),
                cache=lazy_cache,
                length=len(out),
                form=scalar_form,
            )

            init_jerc = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    0,
                    self.forceStochastic,
                    awkward.values_astype(out_dict[jer_name_map["ptGenJet"]], numpy.float32),
                    awkward.values_astype(out_dict[jer_name_map["JetPt"]], numpy.float32),
                    awkward.values_astype(out_dict[jer_name_map["JetEta"]], numpy.float32),
                    awkward.values_astype(out_dict["jet_energy_resolution"], numpy.float32),
                    awkward.values_astype(out_dict["jet_resolution_rand_gauss"], numpy.float32),
                    awkward.values_astype(out_dict["jet_energy_resolution_scale_factor"], numpy.float32),
                ),
                cache=lazy_cache,
            )
            out_dict["jet_energy_resolution_correction"] = init_jerc(length=len(out), form=scalar_form)

            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(out_dict["jet_energy_resolution_correction"], out_dict[jer_name_map["JetPt"]]),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(out_dict["jet_energy_resolution_correction"], out_dict[jer_name_map["JetMass"]]),
                cache=lazy_cache,
            )

            out_dict[self.name_map["JetPt"]] = init_pt_jer(length=len(out), form=scalar_form)
            out_dict[self.name_map["JetMass"]] = init_mass_jer(length=len(out), form=scalar_form)

            out_dict[self.name_map["JetPt"] + "_jer"] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["JetMass"] + "_jer"] = out_dict[self.name_map["JetMass"]]

            # JER systematics
            jerc_up = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    1,
                    self.forceStochastic,
                    awkward.values_astype(out_dict[jer_name_map["ptGenJet"]], numpy.float32),
                    awkward.values_astype(out_dict[jer_name_map["JetPt"]], numpy.float32),
                    awkward.values_astype(out_dict[jer_name_map["JetEta"]], numpy.float32),
                    awkward.values_astype(out_dict["jet_energy_resolution"], numpy.float32),
                    awkward.values_astype(out_dict["jet_resolution_rand_gauss"], numpy.float32),
                    awkward.values_astype(out_dict["jet_energy_resolution_scale_factor"], numpy.float32),
                ),
                cache=lazy_cache,
            )
            up = awkward.flatten(jets)
            up["jet_energy_resolution_correction"] = jerc_up(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            up[self.name_map["JetPt"]] = init_pt_jer(length=len(out), form=scalar_form)
            up[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            jerc_down = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    2,
                    self.forceStochastic,
                    awkward.values_astype(out_dict[jer_name_map["ptGenJet"]], numpy.float32),
                    awkward.values_astype(out_dict[jer_name_map["JetPt"]], numpy.float32),
                    awkward.values_astype(out_dict[jer_name_map["JetEta"]], numpy.float32),
                    awkward.values_astype(out_dict["jet_energy_resolution"], numpy.float32),
                    awkward.values_astype(out_dict["jet_resolution_rand_gauss"], numpy.float32),
                    awkward.values_astype(out_dict["jet_energy_resolution_scale_factor"], numpy.float32),
                ),
                cache=lazy_cache,
            )
            down = awkward.flatten(jets)
            down["jet_energy_resolution_correction"] = jerc_down(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            down[self.name_map["JetPt"]] = init_pt_jer(
                length=len(out), form=scalar_form
            )
            down[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )
            out_dict["JER"] = awkward.zip(
                {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
            )

        # Apply uncertainties (JES)
        has_junc = self.jec_stack.junc is not None
        if self.tool == "clib":
            has_junc = len(self.jec_stack.jec_uncsources_clib) > 0

        if has_junc:
            junc_name_map = dict(self.name_map)
            if has_jer:
                junc_name_map["JetPt"] = junc_name_map["JetPt"] + "_jer"
                junc_name_map["JetMass"] = junc_name_map["JetMass"] + "_jer"
            else:
                junc_name_map["JetPt"] = junc_name_map["JetPt"] + "_jec"
                junc_name_map["JetMass"] = junc_name_map["JetMass"] + "_jec"

            if self.tool == "jecstack":
                junc_args = {
                    k: out_dict[junc_name_map[k]] for k in self.jec_stack.junc.signature
                }
                juncs = self.jec_stack.junc.getUncertainty(**junc_args)

            elif self.tool == "clib":
                junc_out_parms = out.layout.parameters
                junc_out_parms["corrected"] = True
                junc_out = awkward.zip(
                    out_dict, depth_limit=1, parameters=junc_out_parms, behavior=out.behavior
                )
                juncjets = wrap(junc_out)

                uncnames, uncvalues = [], []
                for junc_name in self.jec_stack.jec_uncsources_clib:
                    sf = self.corrections[junc_name]
                    if sf is None:
                        raise ValueError(f"Correction {junc_name} not found in self.corrections")

                    inputs = get_corr_inputs(jets=juncjets, corr_obj=sf, name_map=junc_name_map)
                    unc = awkward.values_astype(sf.evaluate(*inputs), numpy.float32)
                    central = awkward.ones_like(out_dict[self.name_map["JetPt"]])
                    unc_up = central + unc
                    unc_down = central - unc
                    uncnames.append(junc_name.split("_")[-2])
                    uncvalues.append([unc_up, unc_down])
                del juncjets

                # Combine the up and down values into pairs
                combined_uncvalues = [
                    awkward.Array([[up, down] for up, down in zip(unc_up, unc_down)])
                    for unc_up, unc_down in uncvalues
                ]

                juncs = zip(uncnames, combined_uncvalues)

            def junc_smeared_val(uncvals, up_down, variable):
                return awkward.materialized(uncvals[:, up_down] * variable)

            def build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, updown):
                var_dict = dict(in_dict)
                var_dict[jetpt] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        awkward.to_numpy(awkward.values_astype(unc, numpy.float32)),
                        updown,
                        jetpt_orig,
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                var_dict[jetmass] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        awkward.to_numpy(awkward.values_astype(unc, numpy.float32)),
                        updown,
                        jetmass_orig,
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                return awkward.zip(var_dict, depth_limit=1, parameters=out.layout.parameters, behavior=out.behavior)

            def build_variant(unc, jetpt, jetpt_orig, jetmass, jetmass_orig):
                up = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 0)
                down = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 1)
                return awkward.zip({"up": up, "down": down}, depth_limit=1, with_name="JetSystematic")

            for name, func in juncs:
                out_dict[f"jet_energy_uncertainty_{name}"] = func
                out_dict[f"JES_{name}"] = build_variant(
                    func,
                    self.name_map["JetPt"],
                    out_dict[junc_name_map["JetPt"]],
                    self.name_map["JetMass"],
                    out_dict[junc_name_map["JetMass"]],
                )

        out_parms = out.layout.parameters
        out_parms["corrected"] = True
        out = awkward.zip(out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior)

        return wrap(out)




from dataclasses import dataclass, field
from typing import List, Dict, Optional
from coffea.jetmet_tools.FactorizedJetCorrector import FactorizedJetCorrector, _levelre
from coffea.jetmet_tools.JetResolution import JetResolution
from coffea.jetmet_tools.JetResolutionScaleFactor import JetResolutionScaleFactor
from coffea.jetmet_tools.JetCorrectionUncertainty import JetCorrectionUncertainty
import correctionlib as clib

@dataclass
class JECStack:
    """Handles both JEC and clib cases with conditional attributes."""
    # Common fields for both scenarios
    corrections: Dict[str, any] = field(default_factory=dict)
    use_clib: bool = False  # Set to True if useclib is needed

    # Fields for the clib scenario (useclib=True)
    jec_tag: Optional[str] = None
    jec_levels: Optional[List[str]] = field(default_factory=list)
    jer_tag: Optional[str] = None
    jet_algo: Optional[str] = None
    junc_types: Optional[List[str]] = field(default_factory=list)
    json_path: Optional[str] = None
    savecorr: bool = False

    # Fields for the usejecstack scenario (useclib=False)
    jec: Optional[FactorizedJetCorrector] = None
    junc: Optional[JetCorrectionUncertainty] = None
    jer: Optional[JetResolution] = None
    jersf: Optional[JetResolutionScaleFactor] = None


    def __post_init__(self):
        """Handle initialization based on use_clib flag."""
        if self.use_clib:
            self._initialize_clib()
        else:
            self._initialize_jecstack()

    def _initialize_clib(self):
        """Initialize the clib-based correction tools."""
        if not self.json_path:
            raise ValueError("json_path is required for clib initialization.")

        # Load corrections directly from the JSON path
        self.cset = clib.CorrectionSet.from_file(self.json_path)

        # Construct lists for jec, jer, and uncertainties
        self.jec_names_clib = [f"{self.jec_tag}_{level}_{self.jet_algo}" for level in self.jec_levels]
        self.jer_names_clib = []
        self.jec_uncsources_clib = []

        if self.jer_tag is not None:
            self.jer_names_clib = [
                f"{self.jer_tag}_ScaleFactor_{self.jet_algo}",
                f"{self.jer_tag}_PtResolution_{self.jet_algo}"
            ]

        if self.junc_types:
            self.jec_uncsources_clib = [f"{self.jec_tag}_{junc_type}_{self.jet_algo}" for junc_type in self.junc_types]

        # Combine requested corrections
        requested_corrections = self.jec_names_clib + self.jer_names_clib + self.jec_uncsources_clib
        available_corrections = list(self.cset.keys())
        missing_corrections = [name for name in requested_corrections if name not in available_corrections]

        if missing_corrections:
            raise ValueError(
                f"\nMissing corrections in the CorrectionSet: {missing_corrections}. "
                f"\n\nAvailable corrections are: {available_corrections}. "
                f"\n\nRequested corrections are: {requested_corrections}"
            )

        # Store corrections directly in the JECStack for easy access
        self.corrections = {name: self.cset[name] for name in requested_corrections}

    def _initialize_jecstack(self):
        """Initialize the JECStack tools for the non-clib scenario."""
        assembled = self.assemble_corrections()

        if len(assembled["jec"]) > 0:
            self.jec = FactorizedJetCorrector(**assembled["jec"])
        if len(assembled["junc"]) > 0:
            self.junc = JetCorrectionUncertainty(**assembled["junc"])
        if len(assembled["jer"]) > 0:
            self.jer = JetResolution(**assembled["jer"])
        if len(assembled["jersf"]) > 0:
            self.jersf = JetResolutionScaleFactor(**assembled["jersf"])

        if (self.jer is None) != (self.jersf is None):
            raise ValueError("Cannot apply JER-SF without an input JER, and vice-versa!")

    def to_list(self):
        """Convert to list for clib case."""
        return self.jec_names_clib + self.jer_names_clib + self.jec_uncsources_clib + [self.json_path, self.savecorr]

    def assemble_corrections(self):
        """Assemble corrections for both scenarios."""
        assembled = {"jec": {}, "junc": {}, "jer": {}, "jersf": {}}

        for key in self.corrections.keys():
            if "Uncertainty" in key:
                assembled["junc"][key] = self.corrections[key]
            elif ("ScaleFactor" in key or "SF" in key):
                assembled["jersf"][key] = self.corrections[key]
            elif "Resolution" in key and not ("ScaleFactor" in key or "SF" in key):
                assembled["jer"][key] = self.corrections[key]
            elif len(_levelre.findall(key)) > 0:
                assembled["jec"][key] = self.corrections[key]
            else:
                print(f"Unknown correction type for key: {key}")

        return assembled

    @property
    def blank_name_map(self):
        """Returns a blank name map for corrections."""
        out = {
            "massRaw",
            "ptRaw",
            "JetMass",
            "JetPt",
            "METpt",
            "METphi",
            "JetPhi",
            "UnClusteredEnergyDeltaX",
            "UnClusteredEnergyDeltaY",
        }
        if self.jec is not None:
            for name in self.jec.signature:
                out.add(name)
        if self.junc is not None:
            for name in self.junc.signature:
                out.add(name)
        if self.jer is not None:
            for name in self.jer.signature:
                out.add(name)
        if self.jersf is not None:
            for name in self.jersf.signature:
                out.add(name)
        return {name: None for name in out}