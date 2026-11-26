# Simple script to calculate pearson and spearman correlation using metric summary files

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import csv


files = {
    'lex': '../results/lex/lex_summary.csv',
    'sem': '../results/sem/sem_summary.csv',
    'syn': '../results/syn/syn_summary.csv'
}

with open('../results/correlation.csv', 'w') as fout:
    csvwriter = csv.writer(fout)
    csvwriter.writerow(['metric','series','algo','correlation','p-value'])

    # TODO sort data points by author, cannot rely on DictReader ordering
    for key, file in files.items():
        with open(file) as csvfile:
            csvreader = csv.DictReader(csvfile)
            human = []
            sonnet = []
            gemini = []

            for row in csvreader:
                if row['label'] == 'scifi' or row['label'] == 'non-scifi':
                    continue

                if row['group'] == 'human':
                    human.append(row['mean_metric'])
                elif row['group'] == 'sonnet5-5':
                    sonnet.append(row['mean_metric'])
                elif row['group'] == 'gemini-4-hydrogen':
                    gemini.append(row['mean_metric'])

            spearman_sonnet = spearmanr(human, sonnet)
            spearman_gemini = spearmanr(human, gemini)
            csvwriter.writerow([key, 'human-sonnet', 'spearman',
                                round(spearman_sonnet.statistic, 3), 
                                round(spearman_sonnet.pvalue, 3)])
            csvwriter.writerow([key, 'human-gemini', 'spearman',
                                round(spearman_gemini.statistic, 3), 
                                round(spearman_gemini.pvalue, 3)])

            pearson_sonnet = pearsonr(human, sonnet)
            pearson_gemini = pearsonr(human, gemini)
            csvwriter.writerow([key, 'human-sonnet', 'pearson',
                                round(pearson_sonnet.statistic, 3), 
                                round(pearson_sonnet.pvalue, 3)])
            csvwriter.writerow([key, 'human-gemini', 'pearson', 
                                round(pearson_gemini.statistic, 3), 
                                round(pearson_gemini.pvalue, 3)])


