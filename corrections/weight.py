# import pandas as pd
import numpy as np
import awkward as ak
from copy import deepcopy

class Weights(object):
    """
    NOTE: only supports keeping track of nominal weights for now. 
    No ..off or up or down due to issues with deepcopying ak arrays 
    """
    def __init__(self, ones):
        """
        ones -> 1D array representing the eventwise of weights.
        We assume that the event filtering has been done
        """
        self.weights = {
            "nominal" : ones
        }
        self.wgts = {}
        self.variations = []

    def add_weight(self, name, wgt=None, how="nom"):
        if (wgt is None) and ("dummy" not in how):
            return
        # print(f" add_weight name: {name}") #btag_wgt_lfstats1

        # print(f" add_weight wgt: \n {wgt}") #  {'up': pd df, 'down': pd df}
        # print(f"weights add_weight how: {how}")
        if how == "nom":
            # add only nominal weight
            self.add_nom_weight(name, wgt)
            
        elif how == "all":
            # add weight with up and down variations
            nom = wgt["nom"]
            up = wgt["up"]
            down = wgt["down"]
            print(f"add_weight all how nom: \n {nom}")
            self.add_weight_with_variations(name, nom, up, down)
        elif how == "only_vars":
            # add only variations
            up = wgt["up"]
            down = wgt["down"]
            self.add_only_variations(name, up, down)

        elif (how == "dummy_nom") or (how == "dummy"):
            self.add_dummy_weight(name, nom=True, variations=False)
        elif how == "dummy_all":
            self.add_dummy_weight(name, nom=True, variations=True)
        elif how == "dummy_vars":
            self.add_dummy_weight(name, nom=False, variations=True)

        # print(f"add_weight sself.wgts : \n {self.wgts}")
    
    def add_nom_weight(self, name, wgt):
        w_names = self.weights.keys()
        # print(f"weights name_off: {name}_off")
        # print(f'type(self.weights["nominal"]): {type(self.weights["nominal"])}')
        # self.weights[f"{name}_off"] = ak.copy(self.weights["nominal"]) # assign the value of the nominal b4 we overwrite it
        # self.weights[f"{name}_off"] = self.weights["nominal"]

        for w_name in w_names:
            self.weights[w_name] = self.weights[w_name]*wgt
        # self.df[w_name] = (
        #     self.df[columns].multiply(np.array(wgt), axis=0).astype(np.float64)
        # )
        
        self.variations.append(name)
        self.wgts[name] = wgt 

    

    def add_weight_with_variations(self, name, nom_wgt, up, down):
        columns = self.df.columns
        self.wgts[name] = nom_wgt
        nom = self.df["nominal"]
        self.df[f"{name}_off"] = nom
        self.df[f"{name}_up"] = (nom * up).astype(np.float64)
        self.df[f"{name}_down"] = (nom * down).astype(np.float64)
        self.df[columns] = self.df[columns].multiply(nom_wgt, axis=0).astype(np.float64)
        self.variations.append(name)

    def add_only_variations(self, name, up, down):
        nom = self.df["nominal"]
        self.df[f"{name}_up"] = (nom * up).astype(np.float64)
        self.df[f"{name}_down"] = (nom * down).astype(np.float64)
        self.variations.append(name)

    def add_dummy_weight(self, name, nom=True, variations=False):
        self.variations.append(name)
        if nom:
            self.df[f"{name}_off"] = self.df["nominal"]
            self.wgts[name] = 1.0
        if variations:
            self.df[f"{name}_up"] = np.nan
            self.df[f"{name}_down"] = np.nan
            self.df[f"{name}_up"] = self.df[f"{name}_up"].astype(np.float64)
            self.df[f"{name}_down"] = self.df[f"{name}_down"].astype(np.float64)

    def get_weight(self, name, mask=np.array([])):
        if len(mask) == 0:
            mask = np.ones(self.df.shape[0], dtype=bool)
        if name in self.df.columns:
            return self.df[name].to_numpy()[mask]
        else:
            return np.array([])

    def effect_on_normalization(self, mask=np.array([])):
        if len(mask) == 0:
            mask = np.ones(self.df.shape[0], dtype=int)
        for var in self.variations:
            if f"{var}_off" not in self.df.columns:
                continue
            wgt_off = self.df[f"{var}_off"].dropna().to_numpy().sum()
            wgt_on = self.df["nominal"].dropna().to_numpy().sum()
            effect = (wgt_on - wgt_off) / wgt_on * 100
            if effect < 0:
                ef = round(-effect, 2)
                print(f"Enabling {var} decreases yield by {ef}%")
            else:
                ef = round(effect, 2)
                print(f"Enabling {var} increases yield by {ef}%")


    def get_info(self):
        info_str = ""
        for w_name in self.weights.keys():
            info_str += f"{w_name} : \n {ak.to_numpy(self.weights[w_name])} \n"
        return info_str