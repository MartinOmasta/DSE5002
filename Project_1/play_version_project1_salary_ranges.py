# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 01:46:37 2025

@author: Marty
"""

#Section 1: import key packages & libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, t
import types
from matplotlib.table import Table
#END 1#########################################################################



#Section 2: read .csv of salary ranges
infile = "MartinOmasta.module05RProject.csv"
salary_ranges_df = pd.read_csv(infile)
#print(salary_ranges_df.head())
#END 2#########################################################################



#Section 6: Boxplot of what the salary ranges are by experience level
#plt.figure(figsize=(10, 8))
#new_experience_labels = ['Entry', 'Intermediate', 'Senior', 'Executive']

salaries_by_experience_boxplot = sns.boxplot(salary_ranges_df, 
    x = 'experience_level', y = "salary_in_usd", 
    hue = 'experience_level', orient = "v")

salaries_by_experience_boxplot.set(
    title = "USD Salary at Experience Level",
    xlabel = "Experience Level",
    ylabel = "Salary ($ Dollars)")#.set_xticklabels(new_labels)
        

# This ensures a tick mark exists for every label, & creates the label
#num_exp_categories = len(new_experience_labels)
#boxplot_salary_range_by_experience.set_xticks(range(num_exp_categories))
#boxplot_salary_range_by_experience.set_xticklabels(new_experience_labels)

#plt.tight_layout()
#plt.show()
#END 6#########################################################################