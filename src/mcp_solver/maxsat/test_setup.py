import sys


def main():
    """Test the MaxSAT setup."""
    try:
        import pysat
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF

        print("MaxSAT dependencies check: OK")

        # Test creating a basic MaxSAT problem and solving it
        wcnf = WCNF()

        # Add a hard clause: x1 OR x2
        wcnf.append([1, 2])

        # Add soft clauses with weights
        wcnf.append([1], weight=1)  # Prefer x1=True (weight 1)
        wcnf.append([2], weight=2)  # Prefer x2=True (weight 2)

        # Create and use the RC2 solver
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            cost = rc2.cost

            if model:
                print(f"MaxSAT solver test: OK (found model with cost {cost})")
                print(f"Solution: {model}")
            else:
                print("MaxSAT solver test: FAILED (no model found)")
                return 1

        return 0

    except ImportError as e:
        print(f"MaxSAT dependency missing: {e}")
        print("Please install with: uv pip install -e '.[pysat]'")
        return 1
    except Exception as e:
        print(f"MaxSAT setup test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
