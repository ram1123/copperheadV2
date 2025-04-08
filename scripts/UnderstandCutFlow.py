from coffea import processor
from coffea.analysis_tools import PackedSelection, Cutflow
import awkward as ak
from hist import Hist
import vector

vector.register_awkward()

class NanoAODProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            'cutflow': Cutflow(),
            'Z_mass': Hist.new.Reg(30, 70, 110, name="Z_mass").Double(),
            'Z_p': Hist.new.Reg(50, 0, 100, name="Z_p").Double(),
            'Recoil_mass': Hist.new.Reg(20, 100, 200, name="Recoil_mass").Double()
        }
        self.output['cutflow'].add("all_events")
        self.output['cutflow'].add("two_muons")
        self.output['cutflow'].add("Z_mass_window")

    def process(self, events):
        output = self.output.copy()

        Muons = ak.zip({
            'pt': events.Muon_pt,
            'eta': events.Muon_eta,
            'phi': events.Muon_phi,
            'mass': events.Muon_mass,
        }, with_name='PtEtaPhiMLorentzVector')

        two_muons = ak.num(Muons) >= 2
        output['cutflow'].add("all_events", len(events))
        output['cutflow'].add("two_muons", ak.sum(two_muons))

        # Reconstruct Z bosons from muon pairs
        di_muons = ak.combinations(Muons, 2)
        Z = di_muons['0'] + di_muons['1']
        Z_mass_window = (Z.mass > 70) & (Z.mass < 110)
        output['cutflow'].add("Z_mass_window", ak.sum(Z_mass_window))

        # Fill histograms
        selected_Z = Z[Z_mass_window]
        output['Z_mass'].fill(selected_Z.mass)
        output['Z_p'].fill(selected_Z.p)

        return output

    def postprocess(self, accumulator):
        return accumulator


fileset = {'mydata': ['/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018A/5FC505BA-F85E-C343-AA90-3FA2D41F4B43_NanoAOD.root']}

from coffea import processor
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema

# Assuming you have a processor defined somewhere as MyProcessor
class MyProcessor(processor.ProcessorABC):
    # Define your processor similar to earlier examples

# Setting up fileset, assuming you're using NanoAOD files
fileset = {
    'Dataset': ['path/to/your/nanoaod/files.root']
}

# Create a Runner
runner = processor.Runner(
    executor=processor.IterativeExecutor(),  # You can also use FuturesExecutor for parallel processing
    schema=NanoAODSchema,  # Define the schema, assuming NanoAOD
)

# Execute the processor
output = runner(fileset, 'Events', MyProcessor())

print(output)
