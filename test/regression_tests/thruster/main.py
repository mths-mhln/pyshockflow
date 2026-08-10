from pyshockflow import Driver, Config
import sys
import traceback

def main():
    try:
        driver = Driver(configFilePath = "input.ini")
        driver.solve()
        return 0  # success
    except Exception:
        traceback.print_exc()
        return 1  # failure

if __name__ == "__main__":
    sys.exit(main())