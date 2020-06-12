import subprocess
import argparse

iteration = 3000

subprocess.run(['scp',
                f"byrdof@longleaf.unc.edu:/pine/scr/b/y/byrdof/ResearchTestingBed/pre_trained/weight_saves/encoder_{iteration}",
                "pre_trained/weight_saves/."])
subprocess.run(['scp',
                f"byrdof@longleaf.unc.edu:/pine/scr/b/y/byrdof/ResearchTestingBed/pre_trained/weight_saves/decoder_{iteration}",
                "pre_trained/weight_saves/."])
