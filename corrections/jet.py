from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.lookup_tools import extractor




def jec_names_and_sources(jec_pars):
    names = {}
    suffix = {
        "jec_names": [f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_mc"]],
        "jec_names_data": [
            f"_{level}_AK4PFchs" for level in jec_pars["jec_levels_data"]
        ],
        "junc_names": ["_Uncertainty_AK4PFchs"],
        "junc_names_data": ["_Uncertainty_AK4PFchs"],
        "junc_sources": ["_UncertaintySources_AK4PFchs"],
        "junc_sources_data": ["_UncertaintySources_AK4PFchs"],
        "jer_names": ["_PtResolution_AK4PFchs"],
        "jersf_names": ["_SF_AK4PFchs"],
    }

    for key, suff in suffix.items():
        if "data" in key:
            names[key] = {}
            for run in jec_pars["runs"]:
                for tag, iruns in jec_pars["jec_data_tags"].items():
                    if run in iruns:
                        names[key].update({run: [f"{tag}{s}" for s in suff]})
        else:
            tag = jec_pars["jer_tags"] if "jer" in key else jec_pars["jec_tags"]
            names[key] = [f"{tag}{s}" for s in suff]

    return names


def jec_weight_sets(jec_pars):
    weight_sets = {}
    names = jec_names_and_sources(jec_pars)

    extensions = {
        "jec_names": "jec",
        "jer_names": "jr",
        "jersf_names": "jersf",
        "junc_names": "junc",
        "junc_sources": "junc",
    }

    weight_sets["jec_weight_sets_mc"] = []
    weight_sets["jec_weight_sets_data"] = []

    for opt, ext in extensions.items():
        # MC
        weight_sets["jec_weight_sets_mc"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in names[opt]]
        )
        # Data
        if "jer" in opt:
            continue
        data = []
        for run, items in names[f"{opt}_data"].items():
            data.extend(items)
        data = list(set(data))
        weight_sets["jec_weight_sets_data"].extend(
            [f"* * data/jec/{name}.{ext}.txt" for name in data]
        )

    # return weight_sets
    return (weight_sets, names)


def get_name_map(stack):
    name_map = stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"
    return name_map

# def get_jec_factories(jec_parameters: dict, year: str):
def get_jec_factories(jec_parameters: dict):
    # jec_pars = {k: v[year] for k, v in jec_parameters.items()}
    # jec_pars = {k: v for k, v in jec_parameters.items()}
    jec_pars = jec_parameters


    # print(f"jec_factories jec_pars: {jec_pars}")
    # weight_sets = jec_weight_sets(jec_pars)
    # names = jec_names_and_sources(jec_pars)
    weight_sets, names = jec_weight_sets(jec_pars)
    # print(f"jec_factories weight_sets: {weight_sets}")
    # print(f"jec_factories names: {names}")
    jec_factories = {}
    jec_factories_data = {}

    # Prepare evaluators for JEC, JER and their systematics
    jetext = extractor()
    jetext.add_weight_sets(weight_sets["jec_weight_sets_mc"])
    jetext.add_weight_sets(weight_sets["jec_weight_sets_data"])
    jetext.finalize()
    jet_evaluator = jetext.make_evaluator()

    stacks_def = {
        "jec_stack": ["jec_names"],
        "jer_stack": ["jer_names", "jersf_names"],
        "junc_stack": ["junc_names"],
    }

    stacks = {}
    for key, vals in stacks_def.items():
        stacks[key] = []
        for v in vals:
            stacks[key].extend(names[v])
    print(f"jec_factories stacks: \n{stacks}")

    jec_input_options = {}
    jet_variations = ["jec", "junc", "jer"]
    
    for variation in jet_variations:
        # jec_input_options[variation] = {
        #     name: jet_evaluator[name] for name in stacks[f"{variation}_stack"]
        # }
        """
        matches names specific for jet variation with the appropriate jet evaluator
        """
        # jec_input_options[opt] = {
        #     name: jet_evaluator[name] for name in stacks[f"{opt}_stack"]
        # }
        jec_input_options[variation] ={}
        for name in stacks[f"{variation}_stack"]:
            print(f"jec_factories stack name: {name}")
            jec_input_options[variation][name] =jet_evaluator[name] 
        
    print(f"jec_factories jec_input_options: \n {jec_input_options}")
    for src in names["junc_sources"]:
        for key in jet_evaluator.keys():
            if src in key:
                jec_input_options["junc"][key] = jet_evaluator[key]

    # Create separate factories for JEC, JER, JEC variations
    for variation in jet_variations:
        print(f"jec_factories variation: {variation}")
        print(f"jec_factories jec_input_options[variation]: {jec_input_options[variation]}")
        
        stack = JECStack(jec_input_options[variation])
        jec_factories[variation] = CorrectedJetsFactory(get_name_map(stack), stack)

    # Create a separate factory for each data run
    for run in jec_pars["runs"]:
        jec_inputs_data = {}
        for opt in ["jec", "junc"]:
            jec_inputs_data.update(
                {name: jet_evaluator[name] for name in names[f"{opt}_names_data"][run]}
            )
        for src in names["junc_sources_data"][run]:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_inputs_data[key] = jet_evaluator[key]

        jec_stack_data = JECStack(jec_inputs_data)
        jec_factories_data[run] = CorrectedJetsFactory(
            get_name_map(jec_stack_data), jec_stack_data
        )

    return jec_factories, jec_factories_data