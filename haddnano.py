#!/usr/bin/env python3
import ROOT
import numpy
import sys

if len(sys.argv) < 3:
    print("Syntax: haddnano.py out.root input1.root input2.root ...")
    sys.exit(1)
ofname = sys.argv[1]
files = sys.argv[2:]

def zeroFill(tree, brName, brObj, allowNonBool=False):
    # typename: (numpy dtype code, ROOT leaflist type code)
    branch_type_dict = {
        'Bool_t':    ('?', 'O'),
        'Float_t':   ('f4', 'F'),
        'Double_t':  ('f8', 'D'),
        'Int_t':     ('i4', 'i'),
        'UInt_t':    ('u4', 'I'),
        'Long64_t':  ('i8', 'L'),
        # 'ULong64_t': ('u8', 'l'),  # uncomment if needed
    }
    leaf = brObj.GetLeaf(brName)
    if leaf is None:
        raise RuntimeError(f"Leaf for branch {brName} not found in {tree.GetName()}")
    brType = leaf.GetTypeName()
    if (not allowNonBool) and (brType != "Bool_t"):
        raise RuntimeError(f"Did not expect to backfill non-boolean branch {brName} of type {brType}")
    if brType not in branch_type_dict:
        raise RuntimeError(f"Cannot backfill branch of type {brType}")
    buff = numpy.zeros(1, dtype=numpy.dtype(branch_type_dict[brType][0]))
    b = tree.Branch(brName, buff, f"{brName}/{branch_type_dict[brType][1]}")
    # keep basket size reasonable to avoid excessive memory use
    b.SetBasketSize(128 * 1024)
    for _ in range(tree.GetEntries()):
        b.Fill()
    b.ResetAddress()

fileHandles = []
goFast = True
for fn in files:
    print("Adding file", fn)
    fh = ROOT.TFile.Open(fn)
    if not fh or fh.IsZombie():
        raise RuntimeError(f"Could not open input file: {fn}")
    fileHandles.append(fh)
    if len(fileHandles) > 1 and fileHandles[-1].GetCompressionSettings() != fileHandles[0].GetCompressionSettings():
        goFast = False
        print("Disabling fast merging as inputs have different compressions")

of = ROOT.TFile(ofname, "recreate")
if goFast:
    of.SetCompressionSettings(fileHandles[0].GetCompressionSettings())
of.cd()

for e in fileHandles[0].GetListOfKeys():
    name = e.GetName()
    print("Merging", name)
    obj = e.ReadObj()
    inputs = ROOT.TList()
    isTree = obj.IsA().InheritsFrom(ROOT.TTree.Class())
    if isTree:
        obj = obj.CloneTree(-1, "fast" if goFast else "")
        branchNames = set(x.GetName() for x in obj.GetListOfBranches())
    for fh in fileHandles[1:]:
        k = fh.GetListOfKeys().FindObject(name)
        if not k:
            print(f"Warning: key '{name}' missing in {fh.GetName()}, skipping this file for this object")
            continue
        otherObj = k.ReadObj()
        inputs.Add(otherObj)
        if isTree and obj.GetName() in ('Events', 'Runs'):
            otherObj.SetAutoFlush(0)
            otherBranches = set(x.GetName() for x in otherObj.GetListOfBranches())
            missingBranches = list(branchNames - otherBranches)
            additionalBranches = list(otherBranches - branchNames)
            if missingBranches:
                print("missing:", missingBranches, "\n Additional:", additionalBranches)
            allow_nb = (obj.GetName() == 'Runs')
            for br in missingBranches:
                zeroFill(otherObj, br, obj.GetListOfBranches().FindObject(br), allowNonBool=allow_nb)
            for br in additionalBranches:
                branchNames.add(br)
                zeroFill(obj, br, otherObj.GetListOfBranches().FindObject(br), allowNonBool=allow_nb)
        if isTree:
            obj.Merge(inputs, "fast" if goFast else "")
            inputs.Clear()

    if isTree:
        obj.Write()
    elif obj.IsA().InheritsFrom(ROOT.TH1.Class()):
        obj.Merge(inputs)
        obj.Write()
    elif obj.IsA().InheritsFrom(ROOT.TObjString.Class()):
        for st in inputs:
            if st.GetString() != obj.GetString():
                print("Strings are not matching")
        obj.Write()
    else:
        print("Cannot handle", obj.IsA().GetName())
