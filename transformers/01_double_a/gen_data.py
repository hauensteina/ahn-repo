
import argparse
import os
import re
import torch

LETTERS = 'ABCDEFGHIJ'

def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Generate training data for a transformer to rewrite character sequences

    Synopsis:
      {name} [--problem double_a] [--num_samples <int>] [--max_len <int>] outfile
    Description:
        Generate training data for a transformer to rewrite character sequences.

        --type: Which problem to generate data for.  Currently only double_a (double the letter A) is supported.
        --num_samples: How many samples to generate
        --max_len: Maximum length of a sample

    Example:
      python {name} --problem double_a --num_samples 100000 --max_len 100 samples_da.txt

    '''
    msg += '\n '
    return msg

# -------------

def main():
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('outfile', type=str)
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    args = parser.parse_args()
    args = args.__dict__
    run(**args)

def run(outfile, problem, num_samples, max_len):
    if problem == 'double_a':
        samples = gen_double_a_samples(num_samples, max_len)
    else:
        raise Exception(f'Unknown problem: {problem}')
    with open(outfile, 'w') as f:
        for sample in samples:
            f.write(f'{sample}\n')

def gen_double_a_samples(num_samples, max_len):
    samples = []
    for i in range(num_samples):
        sample = gen_double_a_sample(max_len)
        samples.append(sample)
    return samples

def gen_double_a_sample(max_len):
    sample = ''
    samplen = torch.randint(1, max_len, (1,)).item()
    sample = torch.randint(0, len(LETTERS), (samplen,))
    sample = ''.join([LETTERS[i] for i in sample])
    sample_out = re.sub('A', 'AA', sample)
    out = f'{sample},{sample_out}'
    return out

main()
