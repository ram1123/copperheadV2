def print_workspace_vars(ws):
    for var in ws.allVars():
        name = var.GetName()
        val  = var.getVal()
        vmin = var.getMin()
        vmax = var.getMax()
        is_const = var.isConstant()
        print(
            f"{name:20s} = {val:10.5f} "
            f"range = [{vmin:10.5f}, {vmax:10.5f}] "
            f"{'(const)' if is_const else ''}"
        )


def freeze_all_vars(w, make_exception=[]):
    for v in w.allVars():
        do_freeze = True
        name = v.GetName()
        # print(f"name: {name}")
        for exception_name in make_exception:
            if (exception_name in name) or (exception_name == name): # skip
                do_freeze = False
                continue
        if do_freeze:
            v.setConstant(True)