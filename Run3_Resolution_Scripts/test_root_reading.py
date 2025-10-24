import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import dask_awkward as dak


fields_to_load = ["Muon_pt", "Muon_eta", "Muon_phi"]

# fields_to_load = ["pt", "eta", "phi"]

def load(path):
    events = NanoEventsFactory.from_root(
            {path: "Events"},
            schemaclass=NanoAODSchema,
            uproot_options={"timeout":2400},
            metadata={"dataset": "SingleMuon_Run2018C"},
    ).events()

    print(f"fields: {events.fields}")
    print(f"entries: {int(ak.num(events.Muon.pt, axis=0).compute())}")

def loada(path, fields_to_load):
    events = NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=NanoAODSchema,
        uproot_options={"timeout": 2400},
        metadata={"dataset": "SingleMuon_Run2018C"},
    ).events()

    print(f"Available fields: {events.fields}")

    print(f"Muon fields: {events.Muon.fields}")

    events_data = {}
    if fields_to_load:
        for field in fields_to_load:
            events_data[field] = events.Muon[field.replace("Muon_", "")]
            # events_data[f"Muon_{field}"] = events.Muon[field]




    print(f"Loaded fields: {events_data.keys()}")
    print(f"Loaded fields: {events_data}")

    # events_data_ak = ak.zip(events_data)

    # print(f"Entries (Muon_pt): {int(ak.num(events_data_ak.Muon_pt, axis=0).compute())}")


# load_path = "/eos/purdue/store/user/rasharma/customNanoAOD_Others/UL2018/SingleMuon_Run2018C/*.root"
load_path = "/depot/cms/hmm/shar1172/test/*.root"
# events_data = load(load_path)
events_data = loada(load_path, fields_to_load)
