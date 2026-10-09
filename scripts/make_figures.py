"""C18: regenerate every figure that is drawn from MLflow-backed (and, with --derived, locally derived) results.

    python scripts/make_figures.py [--derived]

Same steps and provenance as scripts/make_tables.py (each step writes its tables and figures together), limited to
the steps that produce figures.
"""

from make_tables import main

if __name__ == "__main__":
    main(figures_only=True)
