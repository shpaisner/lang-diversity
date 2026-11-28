# Simple script to calculate pearson and spearman correlation using metric summary files

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import csv


files = {
    'lex': '../results/lex/lex_summary.csv',
    'syn': '../results/syn/syn_summary.csv',
    'sem': '../results/sem/sem_summary.csv',
}

with open('../results/correlation.csv', 'w') as fout:
    csvwriter = csv.writer(fout)
    csvwriter.writerow(['metric','model','correlation','p-value'])

    for key, file in files.items():
        with open(file) as csvfile:
            csvreader = csv.DictReader(csvfile)
            human = []
            sonnet = []
            gemini = []

            for row in csvreader:
                if row['label'] == 'scifi' or row['label'] == 'non-scifi' or ('avg' in row['label']):
                    continue

                point = (row['label'], float(row['mean_metric']))
                if row['group'] == 'human':
                    human.append(point)
                elif row['group'] == 'sonnet5-5':
                    sonnet.append(point)
                elif row['group'] == 'gemini-4-hydrogen':
                    gemini.append(point)

            # sort arrays by author, cannot rely on DictReader to sort
            human.sort(); sonnet.sort(); gemini.sort()
            human_scores  = [m for _, m in human]
            sonnet_scores = [m for _, m in sonnet]
            gemini_scores = [m for _, m in gemini]

            spearman_sonnet = spearmanr(human_scores, sonnet_scores)
            spearman_gemini = spearmanr(human_scores, gemini_scores)
            csvwriter.writerow([key, 'human-sonnet', 'spearman',
                                round(spearman_sonnet.statistic, 3), 
                                round(spearman_sonnet.pvalue, 3)])
            csvwriter.writerow([key, 'human-gemini', 'spearman',
                                round(spearman_gemini.statistic, 3), 
                                round(spearman_gemini.pvalue, 3)])

            pearson_sonnet = pearsonr(human_scores, sonnet_scores)
            pearson_gemini = pearsonr(human_scores, gemini_scores)
            csvwriter.writerow([key, 'sonnet',
                                round(pearson_sonnet.statistic, 3), 
                                round(pearson_sonnet.pvalue, 3)])
            csvwriter.writerow([key, 'gemini',
                                round(pearson_gemini.statistic, 3), 
                                round(pearson_gemini.pvalue, 3)])


