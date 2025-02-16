# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:23:56 2024

@author: 17154
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = r"D:\BaiduSyncdisk\Research\Succolarity\for Publications\Succolarity-Literature.xlsx"
df = pd.read_excel(file_path)

# Clean the 'Fields' column by removing any leading/trailing whitespace and dropping empty fields
df['Fields'] = df['Fields'].str.strip()
df = df[df['Fields'].notnull() & (df['Fields'] != '')]

# Split the 'Fields' column into multiple rows
df_exploded = df.assign(Fields=df['Fields'].str.split(';')).explode('Fields')

# Clean up any whitespace in the 'Fields' column again
df_exploded['Fields'] = df_exploded['Fields'].str.strip()

# Ensure there are no empty strings in 'Fields'
df_exploded = df_exploded[df_exploded['Fields'] != '']

# Count the frequency of each field per year
field_counts = df_exploded.groupby(["Year", "Fields"]).size().reset_index(name="Count")

# Count the number of publications per year
publications_per_year = df.groupby("Year").size().reset_index(name="Publications")

# Generate distinct colors for the research fields
colormap = plt.cm.get_cmap('tab20', len(field_counts["Fields"].unique()))
colors = {field: colormap(i) for i, field in enumerate(field_counts["Fields"].unique())}

# Adjustable Parameters
bubble_scale_factor = 160  
legend_font_size = 20      
legend_title_font_size = 22  
label_font_size = 24       
tick_font_size = 24        
title_font_size = 26       
title_pad = 40             
x_labelpad = 28            
y_labelpad = 24            
legend_spacing = 1.2       

# Plotting
fig, ax1 = plt.subplots(figsize=(21, 15), dpi=600)
fig.subplots_adjust(top=0.5)  

# Set the border color and width for the plot's axes spines
for spine in ax1.spines.values():
    spine.set_edgecolor('black')  
    spine.set_linewidth(1)        

# Bar plot for the number of publications per year
ax1.bar(publications_per_year["Year"], publications_per_year["Publications"], color='gray', alpha=0.5)
ax1.set_xlabel("Year", fontsize=label_font_size, labelpad=x_labelpad)  
ax1.set_ylabel("Number of Publications", color='black', fontsize=label_font_size, labelpad=y_labelpad)  # Adjusting labelpad
ax1.tick_params(axis='y', labelcolor='black', labelsize=tick_font_size)
ax1.tick_params(axis='x', labelsize=tick_font_size)

# Bubble plot for research fields
ax2 = ax1.twinx()
unique_fields = field_counts["Fields"].unique()
field_to_num = {field: num for num, field in enumerate(unique_fields)}

for field in unique_fields:
    subset = field_counts[field_counts["Fields"] == field]
    y_values = [field_to_num[field]] * len(subset)
    ax2.scatter(subset["Year"], y_values, 
                s=subset["Count"] * bubble_scale_factor,  
                alpha=0.6, edgecolors="w", linewidth=2, label=field, color=colors[field])

ax2.set_yticks(list(field_to_num.values()))
ax2.set_yticklabels(list(field_to_num.keys()), fontsize=tick_font_size)


ax1.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()

# Adjust the legend size, title size, and spacing
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), 
           bbox_to_anchor=(0, 1), loc='upper left', 
           fontsize=legend_font_size,  
           title="Research Fields", title_fontsize=legend_title_font_size,  
           ncol=2,
           labelspacing=legend_spacing,  
           handletextpad=1.5,  
           frameon=True,       
           edgecolor='black',  
           framealpha=1,       
           borderpad=1.5)      

output_file = "evolution_of_research_fields1.jpg"
plt.savefig(output_file, format='jpg', dpi=600, bbox_inches='tight')  

plt.show()


