with open("main_run_output.txt", "r") as file:
    lines = file.read().splitlines()
    props = list(set([line.split(":")[1].strip() for line in lines if line.split(":")[0] == "prop"]))
    x_strs = list(set([line.split(":")[1].strip() for line in lines if line.split(":")[0] == "x_str"]))
    y_strs = list(set([line.split(":")[1].strip() for line in lines if line.split(":")[0] == "y_str"]))
    print("unique props:", props)
    print("unique x_strs:", x_strs)
    print("unique y_strs:", y_strs)

