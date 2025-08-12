import dask_awkward as dak
import numpy as np
import awkward as ak
import argparse
import sys
from distributed import LocalCluster, Client, progress
np.set_printoptions(threshold=sys.maxsize)
import correctionlib
from correctionlib import schemav2 as schema


def filterRegion(events, region="h-peak"):
    dimuon_mass = events.dimuon_mass
    if region =="h-peak":
        region = (dimuon_mass > 115.03) & (dimuon_mass < 135.03)
    elif region =="h-sidebands":
        region = ((dimuon_mass > 110) & (dimuon_mass < 115.03)) | ((dimuon_mass > 135.03) & (dimuon_mass < 150))
    elif region =="signal":
        region = (dimuon_mass >= 110) & (dimuon_mass <= 150.0)
    elif region =="z-peak":
        region = (dimuon_mass >= 70) & (dimuon_mass <= 110.0)

    # mu1_pt = events.mu1_pt
    # mu1ptOfInterest = (mu1_pt > 75) & (mu1_pt < 150.0)
    # events = events[region&mu1ptOfInterest]
    events = events[region]
    return events

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()   
    parser.add_argument(
        "--use_gateway",
        dest="use_gateway",
        default=False, 
        action=argparse.BooleanOptionalAction,
        help="If true, uses dask gateway client instead of local",
    )
    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        default="2018",
        action="store",
        help="string value of year we are calculating",
    )
    parser.add_argument(
        "-l",
        "--label",
        dest="label",
        default="test",
        action="store",
        help="Unique run label (to create output path)",
    )
    args = parser.parse_args()
    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        print(f"client: {client}")
        print("Gateway Client found")
    else:
        from distributed import LocalCluster, Client
        cluster = LocalCluster(processes=True)
        cluster.adapt(minimum=8, maximum=31) #min: 8 max: 32
        client = Client(cluster)
        print("Local scale Client created")

    
    # run_label = "V2_Jan09_ForZptReWgt"
    run_label = args.label
    year = args.year
    base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{run_label}/stage1_output/{year}/f1_0" # define the save path of stage1 outputs

    # # temporary overwrite
    # year = "2016"
    # base_path = f"/depot/cms/users/yun79/hmm/copperheadV1clean/{run_label}/stage1_output/2016*/f1_0" # define the save path of stage1 outputs
    
    # load the data and dy samples
    data_events = dak.from_parquet(f"{base_path}/data_*/*/*.parquet")
    dy_events = dak.from_parquet(f"{base_path}/dy_M-50/*/*.parquet")

    # apply z-peak region filter and nothing else
    data_events = filterRegion(data_events, region="z-peak")
    dy_events = filterRegion(dy_events, region="z-peak")
    if "2016" in year:
        pt_bin_edges = np.array([ # binnning optimized by me for 2016 eras
                0.        ,   3.33333333,   6.66666667,  10.        ,
                13.33333333,  16.66666667,  20.        ,  23.33333333,
                26.66666667,  30.        ,  33.33333333,  36.66666667,
                40.        ,  43.33333333,  46.66666667,  50.        ,
                53.33333333,  56.66666667,  60.        ,  70.        , 
                100.        ,  200.  
        ])
    else: # 2017 and 2018
        pt_bin_edges = np.array([ # this is the bin edges that valerie used from old zpt rewgts used for 2018 and 2018
                0.        ,   3.33333333,   6.66666667,  10.        ,
                13.33333333,  16.66666667,  20.        ,  23.33333333,
                26.66666667,  30.        ,  33.33333333,  36.66666667,
                40.        ,  43.33333333,  46.66666667,  50.        ,
                53.33333333,  56.66666667,  60.        ,  63.33333333,
                66.66666667,  70.        ,  73.33333333,  76.66666667,
                80.        ,  93.33333333, 116.66666667, 140.        ,
               163.33333333, 186.66666667, 200.        
        ])
    
    # compute the events to local memory
    njet_field = "njets_nominal"
    value_field = "dimuon_pt"
    weight_field = "wgt_nominal"
    fields2load = [njet_field, value_field, weight_field]
    # dy_events = ak.zip({field: dy_events for field in fields2load}).compute()
    # data_events = ak.zip({field: data_events for field in fields2load}).compute()
    

    SF_hists = []
    for njet in [0,1,2]:
        if njet != 2:
            data_events_loop = data_events[data_events[njet_field] ==njet]
            dy_events_loop = dy_events[dy_events[njet_field] ==njet]
        else:
            data_events_loop = data_events[data_events[njet_field] >=njet]
            dy_events_loop = dy_events[dy_events[njet_field] >=njet]

        
        # data_hist, edges = np.histogram(data_events_loop[value_field], bins=pt_bin_edges, weights=data_events_loop[weight_field])
        # dy_hist, edges = np.histogram(dy_events_loop[value_field], bins=pt_bin_edges, weights=dy_events_loop[weight_field])
        data_hist, edges = np.histogram(data_events_loop[value_field].compute(), bins=pt_bin_edges, weights=data_events_loop[weight_field].compute())
        dy_hist, edges = np.histogram(dy_events_loop[value_field].compute(), bins=pt_bin_edges, weights=dy_events_loop[weight_field].compute())
        SF_hist = data_hist/dy_hist
        SF_hists.append(SF_hist)

        # debugging print
        print(f"njet {njet} data_hist: {ak.to_numpy(data_hist)}")
        print(f"njet {njet} dy_hist: {ak.to_numpy(dy_hist)}")
        print(f"njet {njet} SF_hist: {ak.to_numpy(SF_hist)}")
        # print(f"njet {njet} data_hist: {ak.to_numpy(data_hist)}")
        # print(f"njet {njet} dy_hist: {ak.to_numpy(dy_hist)}")
        # print(f"njet {njet} SF_hist: {', '.join(map(str, SF_hist))}")

    # Define a correction with three input parameters
    correctionlib_content = np.concatenate(SF_hists)
    print(f"correctionlib_content: {correctionlib_content}")
    
    njet_edges = [-0.01, 0.99, 1.99, 50] #njet edges in float format -> njets==0, ==1 or >= 2
    correction = schema.Correction(
        name="Zpt_rewgt",
        description=f"Zpt re-weight for {year}",
        version=1,
        inputs=[
            schema.Variable(name="njets", type="real", description="number of jets"),
            schema.Variable(name="dimuon_pt", type="real", description="dimuon pt"),
        ],
        output=schema.Variable(name="correction_factor", type="real"),
        data=schema.MultiBinning(
            nodetype="multibinning",
            inputs=["njets", "dimuon_pt"],
            edges=[
                njet_edges,
                pt_bin_edges,
                
            ],
            content=correctionlib_content,
            flow="clamp"  # Handles out-of-bounds input values
        )
    )
    
    # Create the correction set and add the correction
    correction_set = schema.CorrectionSet(
        schema_version=schema.VERSION,
        corrections=[correction]
    )
    
    # Save the correction set to a JSON file
    
    json_name = f"ZptReWgt_{year}UL.json"
    
    with open(json_name, "w") as fout:
        fout.write(correction_set.json(exclude_unset=True))


    """
    validate the SF are saved correctly in correctionlib using test cases
    """
    import awkward as ak
    
    # Load the correction set
    correction_set = correctionlib.CorrectionSet.from_file(json_name)
    
    # Access the specific correction by name
    correction = correction_set["Zpt_rewgt"]
    
    # Create an evaluator for the correction
    def evaluate_correction(njet, dimuon_pt):
        return correction.evaluate(njet, dimuon_pt)
    
    # Test the evaluator with some example inputs
    test_data = [
        [35.0, 0.5],     # njet>=2, bin 0
        [35.0, 4],     # njet>=2, bin 1
        [2.0, 210],     # njet>=2, last bin
        [1.0, 0.5],     # njet==1, bin 1
        [1.0, 4],     # njet==1, bin 1
        [1.0, 210],     # njet==1, last bin
        [0.0, 0.5],     # njet==1, bin 1
        [0.0, 4],     # njet==1, bin 1
        [0.0, 210],     # njet==1, last bin
    ]
    test_data = ak.Array(test_data)
    # Evaluate the correction for each input triplet
    results = [evaluate_correction(njet, dimuon_pt) for njet, dimuon_pt in test_data]
    
    # Print results
    for (njet, dimuon_pt), result in zip(test_data, results):
        print(f"njet: {njet}, dimuon_pt: {dimuon_pt}, scale factor: {result}")
    
    
