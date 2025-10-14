#!/usr/bin/env python3
# split_dy_by_gjj_mass.py
import argparse, os, sys
import dask_awkward as dak
import awkward as ak


def main():
    ap = argparse.ArgumentParser(
        description="Split DY parquet into >=2-jet and <2-jet subsets using gjj_mass."
    )
    ap.add_argument(
        "--minnlo", nargs="+", default=["dy_M-100To200_MiNNLO/", "dy_M-50_MiNNLO/"]
    )
    ap.add_argument("--vbff", nargs="+", default=["dy_VBF_filter/"])
    ap.add_argument(
        "--outdir-ge2",
        required=True,
        help="Output directory for gjj_mass >= 0 (>=2 jets)",
    )
    ap.add_argument(
        "--outdir-lt2",
        required=True,
        help="Output directory for gjj_mass < 0 or missing (<2 jets)",
    )
    ap.add_argument(
        "--repartition",
        type=int,
        default=0,
        help="Target number of output partitions (0 = keep original).",
    )
    ap.add_argument(
        "--use_gateway",
        dest="use_gateway",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, uses dask gateway client instead of local",
    )
    args = ap.parse_args()

    if args.use_gateway:
        from dask_gateway import Gateway
        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        # gateway = Gateway()
        print("Connecting to Dask Gateway")
        print(f"gateway: {gateway}")
        print(f"gateway list clusters: {gateway.list_clusters()}")

        cluster_info = gateway.list_clusters()[0] # get the first cluster by default. There only should be one anyways
        client = gateway.connect(cluster_info.name).get_client()
        print(f"client: {client}")
        print("Gateway Client created")
    else:
        from distributed import Client
        client = Client(n_workers=64,  threads_per_worker=1, processes=True, memory_limit='10 GiB')
        print("Local scale Client created")

    # Load datasets
    ds_min = dak.from_parquet(args.minnlo)
    ds_vbf = dak.from_parquet(args.vbff)

    if "gjj_mass" not in ds_min.fields or "gjj_mass" not in ds_vbf.fields:
        print("ERROR: 'gjj_mass' not found in input parquet.", file=sys.stderr)
        sys.exit(1)

    gjj_min = ak.fill_none(ds_min["gjj_mass"], -1)
    gjj_vbf = ak.fill_none(ds_vbf["gjj_mass"], -1)

    # Masks per your rule:
    mask_lt2       = (gjj_min <= 0)                  # 0/1 gen-jet (or missing)  → DYJ01
    mask_ge2_le350 = (gjj_min > 0) & (gjj_min <= 350)    # 2+ gen-jets but mjj ≤ 350 → goes with DY_VBF
    mask_ge2       = (gjj_vbf > 350)                    # 2+ gen-jets and mjj > 0   → DYJ2 from VBF

    dyj01 = ds_min[mask_lt2]
    dyj2_from_min = ds_min[mask_ge2_le350]
    dyj2_from_vbf = ds_vbf[mask_ge2]

    # --- Schema align + concat ---
    common = [f for f in dyj2_from_min.fields if f in dyj2_from_vbf.fields]
    dyj2 = dak.concatenate([dyj2_from_vbf[common], dyj2_from_min[common]], axis=0)

    ds_ge2 = dyj2
    ds_lt2 = dyj01

    # Optional repartitioning for balanced write
    if args.repartition > 0:
        ds_ge2 = dak.repartition(ds_ge2, args.repartition)
        ds_lt2 = dak.repartition(ds_lt2, args.repartition)

    os.makedirs(args.outdir_lt2, exist_ok=True)
    out_ge2_min = os.path.join(args.outdir_ge2, "minnlo")
    out_ge2_vbf = os.path.join(args.outdir_ge2, "vbf")
    os.makedirs(out_ge2_min, exist_ok=True)
    os.makedirs(out_ge2_vbf, exist_ok=True)

    # write each piece independently (avoids concat → PlaceholderArray carry)
    dyj01.to_parquet(args.outdir_lt2)
    dyj2_from_min.to_parquet(out_ge2_min)
    dyj2_from_vbf.to_parquet(out_ge2_vbf)

    print(f"Wrote DYJ01 -> {args.outdir_lt2}")
    print(f"Wrote DYJ2(minnlo, 0<gjj<=350) -> {out_ge2_min}")
    print(f"Wrote DYJ2(vbf, gjj>350 or all if missing) -> {out_ge2_vbf}")

# # Small summary (lazy counts trigger minimal computation)
# n_ge2 = dak.sum(dak.ones_like(ds["gjj_mass"])[mask_ge2]).compute()
# n_lt2 = dak.sum(dak.ones_like(ds["gjj_mass"])[~mask_ge2]).compute()
# print(f"Events written: >=2 jets: {n_ge2} | <2 jets: {n_lt2}")

if __name__ == "__main__":
    main()
