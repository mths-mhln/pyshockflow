from pyshockflow import Driver, Config
import sys
import traceback

def main():
    try:
        config = Config('input.ini')
        driver = Driver(config)
        driver.solve()
        return 0  # success
    except Exception:
        traceback.print_exc()
        return 1  # failure

if __name__ == "__main__":
    sys.exit(main())