import pickle

with open("Results/outletPressure_45kPa_NX_200/Results.pik", "rb") as f:
    results = pickle.load(f)


# dump entire results in a text file
with open("Results/outletPressure_45kPa_NX_200/Results.txt", "w") as f:
    for key, value in results.items():
        if key == "Primitive":
            f.write(f"{key}:\n")
            for var_name, var_values in value.items():
                f.write(f"  {var_name}:\n")
                for i, var_value in enumerate(var_values):
                    f.write(f"    Node {i}: {var_value}\n")
        f.write(f"{key}: {value}\n")

