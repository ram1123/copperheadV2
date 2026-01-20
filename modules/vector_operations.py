import awkward as ak


def convertVectorType4D(vector, vector_name):
    new_vector = ak.zip(
        {
            "pt": vector.pt,
            "eta": vector.eta,
            "phi": vector.phi,
            "mass": vector.mass,
            "charge": vector.charge,
        },
        with_name=vector_name,
        behavior=vector.behavior,
    )
    return new_vector
