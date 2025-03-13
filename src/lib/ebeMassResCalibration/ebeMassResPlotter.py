import dask_awkward as dak
import awkward as ak
from distributed import Client
import time
from basic_class_for_calibration import get_calib_categories
from basic_class_for_calibration import generateBWxDCB_plot
from basic_class_for_calibration import generateVoigtian_plot


if __name__ == "__main__":
    client =  Client(n_workers=5,  threads_per_worker=1, processes=True, memory_limit='10 GiB')
    total_time_start = time.time()
    common_load_path = "/work/users/yun79/stage1_output/Run2StorageTest/2018/f1_0"
    data_load_path = common_load_path+"/data*/*/*.parquet"

    # common_load_path = "/depot/cms/users/shar1172/hmm/copperheadV1clean/Run3_nanoAODv12_BSOff/stage1_output/2022preEE/f1_0"
    # data_load_path = common_load_path+"/data_C/*/*.parquet"
    # data_load_path = common_load_path+"/data_D/*/*.parquet"

    data_events = dak.from_parquet(data_load_path)

    # print entries
    print(data_events)

    # we're only interested in ZCR
    region_filter = ak.fill_none(data_events["z_peak"], value=False)
    data_events = data_events[region_filter]

    # only select specific fields to load to save run time
    fields_of_interest = ["mu1_pt", "mu1_eta", "mu2_eta","dimuon_mass"] # mu1,mu2 are needed to separate categories
    data_events = data_events[fields_of_interest]

    # load data to memory using compute()
    data_events = ak.zip({
        field : data_events[field] for field in data_events.fields
    }).compute()
    data_categories = get_calib_categories(data_events)
    print(f"total number of categories: {len(data_categories)}")
    # print(f"categories keys: {data_categories.keys()}")
    print(f"categories: {data_categories}")

    nbins = 100 # 100

    # iterate over 30 different calibration categories
    total_categories = len(data_categories)
    # for idx in range(total_categories):
    counter = 0
    for key, idx in data_categories.items():
        # print(f"idx: {idx}")
        # sys.exit(0)
        # cat_selection = data_categories[idx]
        cat_selection = idx
        cat_dimuon_mass = ak.to_numpy(data_events.dimuon_mass[cat_selection])
        if counter < 12:
            generateBWxDCB_plot(cat_dimuon_mass, key, nbins=nbins, logfile="CalibrationLog.txt")
        else:
            generateVoigtian_plot(cat_dimuon_mass, key, nbins=nbins, logfile="CalibrationLog.txt")
        counter += 1



    print("Success!")
    print(f"total time elapsed : {time.time() - total_time_start}")
