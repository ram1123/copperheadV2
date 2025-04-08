
def getDatasetRootFiles(single_dataset_name: str, allowlist_sites: list)-> list:
    print(f"single_dataset_name {single_dataset_name}")
    fnames = glob.glob(f"{single_dataset_name}/*.root")
    logger.debug(f"fnames: {fnames}")
    fnames = [fname.replace("/eos/purdue", "root://eos.cms.rcac.purdue.edu/") for fname in fnames] # replace to xrootd bc sometimes eos mounts timeout when reading

    return fnames

def getBadFile(fname):
    try:
        up_file = uproot.open(fname)
        # up_file["Events"]["Muon_pt"].array() # check that you could read branches
        if "Muon_pt" in up_file["Events"].keys():
            return "" # good file
        else:
            return fname # bad file
    except Exception as e:
        # return f"An error occurred with file {fname}: {e}"
        return fname # bad fileclient


def getBadFileParallelizeDask(filelist):
    """
    We assume that the dask client has already been initialized
    """
    lazy_results = []
    for fname in filelist:
        lazy_result = dask.delayed(getBadFile)(fname)
        lazy_results.append(lazy_result)
    results = dask.compute(*lazy_results)

    bad_file_l = []
    for result in results:
        if result != "":
            # print(result)
            bad_file_l.append(result)
    return bad_file_l


def removeBadFiles(filelist):
    bad_filelist = getBadFileParallelizeDask(filelist)
    clean_filelist = list(set(filelist) - set(bad_filelist))
    return clean_filelist

