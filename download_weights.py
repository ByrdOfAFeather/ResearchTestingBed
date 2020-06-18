import subprocess
import argparse

iteration = 48000

subprocess.run(['scp',
                f"byrdof@longleaf.unc.edu:/pine/scr/b/y/byrdof/ResearchTestingBed/pre_trained/weight_saves/encoder_fixed_attn{iteration}",
                "pre_trained/weight_saves/."])
subprocess.run(['scp',
                f"byrdof@longleaf.unc.edu:/pine/scr/b/y/byrdof/ResearchTestingBed/pre_trained/weight_saves/decoder_fixed_attn{iteration}",
                "pre_trained/weight_saves/."])
