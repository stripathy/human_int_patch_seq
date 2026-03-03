library(tidyverse)
library(cowplot)
library(patchwork)
library(forcats) # For reordering the x-axis


# this is the data from Kyle T illustrating the mapping of Allen Institute's internal human Patch-seq data to SEA-AD taxonomy
patch_seq_scanvi = read_csv('~/Github/patch_seq_lee/iterative_scANVI_results_patchseq_only.2022-11-22.csv')
colnames(patch_seq_scanvi)[1] = 'exp_component_name'

# this is the data from the allen institute lee patch-seq dataset (layers 2-6)

# it's organized in two parts, one from acute slices, another from cultured slices
load("~/Github/patch_seq_lee/complete_patchseq_data_sets1.RData")
load("~/Github/patch_seq_lee/complete_patchseq_data_sets2.RData")
datPatch <- cbind(datPatch1,datPatch2)
save(datPatch,annoPatch,metaPatch,file="~/Github/patch_seq_lee/complete_patchseq_data_sets.RData")

lee_dalley_meta = read_csv('~/Github/patch_seq_lee/LeeDalley_manuscript_metadata_v2.csv')
lee_dalley_ephys = read_csv('~/Github/patch_seq_lee/LeeDalley_ephys_fx.csv')
lee_dalley_morpho = read_csv('~/Github/patch_seq_lee/LeeDalley_morpho_features.csv')
lee_dalley_donors = read_csv('~/Github/patch_seq_lee/DataS1_Donor_table.csv')

metaPatch = left_join(metaPatch, lee_dalley_donors)
joined_results = left_join(patch_seq_scanvi, metaPatch)

joined_dataset = left_join(lee_dalley_meta, joined_results %>% dplyr::rename(specimen_id_x = spec_id), by = 'specimen_id_x')


joined_dataset = left_join(joined_dataset, left_join(lee_dalley_ephys, lee_dalley_morpho) %>% dplyr::rename(specimen_id_x = specimen_id), by = 'specimen_id_x')

joined_dataset <- joined_dataset %>%
  mutate(
    age_norm = as.numeric(str_extract(Age, "\\d+"))
  )

plotting_dataset %>% filter(subclass_scANVI %in% c('Sst'), 
         !is.na(Cortical_layer_numeric)) %>%
  ggplot(aes(x = age_norm, y = sag, color = Cortical_layer)) + 
  geom_point() + geom_smooth(method = 'lm', se = F) + theme_cowplot()

data = plotting_dataset %>% filter(subclass_scANVI %in% c('Sst'), 
         !is.na(Cortical_layer_numeric), !is.na(age_norm)) 

m1 = lmer('sag ~ Cortical_layer + scale(age_norm) + (1|Donor.x)', data = data) 

m2 = lmer('sag ~ Cortical_layer  + (1|Donor.x)', data = data) 
anova(m2, m1)


library(ggplot2)
library(dplyr)

# --- 1. Define your vulnerability groups ---
vulnerable_types <- c('Sst_25', 'Sst_22', 'Sst_2', 'Sst_20', 'Sst_3')
ad_extra_vulnerable_types <- c('Sst_19', 'Sst_9', 'Sst_23', 'Sst_11', 'Sst_13')
non_vulnerable_types <- c('Sst_1', 'Sst_4', 'Sst_5', 'Sst_7', 'Sst_10', 'Sst_12' )

# --- 2. Define a custom color palette with NEW names ---
vulnerability_colors <- c(
  "AD Vulnerable"       = "#D55E00", # Vermillion/Red
  "AD+SCZ Vulnerable" = "#E69F00", # Orange
  "Non-vulnerable"    = "grey50",
  "Other"             = "black"
)

# --- 3. Prepare data and create plot ---
plotting_dataset = joined_dataset %>%
  # *** FIX: Create a new numeric layer column FIRST ***
  mutate(
    Cortical_layer_numeric = case_when(
      is.na(Cortical_layer) | Cortical_layer == "NA" ~ NA_real_,
      Cortical_layer == "5_6" ~ 5.5,
      Cortical_layer == "2_3" ~ 2.5,
      # All other cases should be single numbers
      TRUE ~ as.numeric(Cortical_layer) 
    )
  ) %>% 
  
  # Filter for Sst and *clean* layer data
  filter(subclass_scANVI %in% c('Sst'), 
         !is.na(Cortical_layer_numeric)) %>%
  
  # Add vulnerability groups
  mutate(
    vulnerability_group = case_when(
      supertype_scANVI %in% ad_extra_vulnerable_types ~ "AD Vulnerable",
      supertype_scANVI %in% vulnerable_types        ~ "AD+SCZ Vulnerable",
      supertype_scANVI %in% non_vulnerable_types    ~ "Non-vulnerable",
      TRUE                                        ~ "Other"
    ),
    vulnerability_group = factor(vulnerability_group, 
                                 levels = c("AD Vulnerable", "AD+SCZ Vulnerable", "Non-vulnerable", "Other"))
  ) 

  # Pipe the prepared data into ggplot
  # *** FIX: Reorder using the new 'Cortical_layer_numeric' column ***
cortical_layers_plot = plotting_dataset %>% ggplot(aes(x = reorder(supertype_scANVI, Cortical_layer_numeric, mean), 
             y = Cortical_layer_numeric)) +
  
  # Geoms
  geom_boxplot(outlier.shape = NA, fill = "grey95") +
  geom_jitter(
    aes(color = vulnerability_group), 
    width = 0.2, 
    alpha = 0.9
  ) +
  
  # Scales, Labs, and Theme
  scale_color_manual(values = vulnerability_colors) +
  labs(
    y = "Cortical Layer",
    x = "Supertype",
    color = "Vulnerability Group"
  ) + 
  scale_y_reverse() + 
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position = "right"
  )


  # Pipe the prepared data into ggplot
  # *** FIX: Reorder using the new 'Cortical_layer_numeric' column ***
sag_plot = plotting_dataset %>% ggplot(aes(x = reorder(supertype_scANVI, Cortical_layer_numeric, mean), 
             y = sag)) +
  
  # Geoms
  geom_boxplot(outlier.shape = NA, fill = "grey95") +
  geom_jitter(
    aes(color = vulnerability_group), 
    width = 0.2, 
    alpha = 0.9
  ) +
  
  # Scales, Labs, and Theme
  scale_color_manual(values = vulnerability_colors) +
  labs(
    y = "Sag (ratio)",
    x = "Supertype",
    color = "Vulnerability Group"
  ) + 
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position = "right"
  )

  # Pipe the prepared data into ggplot
  # *** FIX: Reorder using the new 'Cortical_layer_numeric' column ***
tau_plot = plotting_dataset %>% ggplot(aes(x = reorder(supertype_scANVI, Cortical_layer_numeric, mean), 
             y = tau * 1000)) +
  
  # Geoms
  geom_boxplot(outlier.shape = NA, fill = "grey95") +
  geom_jitter(
    aes(color = vulnerability_group), 
    width = 0.2, 
    alpha = 0.9
  ) +
  
  # Scales, Labs, and Theme
  scale_color_manual(values = vulnerability_colors) +
  labs(
    y = "Membrane tau (ms)",
    x = "Supertype",
    color = "Vulnerability Group"
  ) + 
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position = "right"
  )

  # Pipe the prepared data into ggplot
  # *** FIX: Reorder using the new 'Cortical_layer_numeric' column ***
up_down_ratio_plot = plotting_dataset %>% ggplot(aes(x = reorder(supertype_scANVI, Cortical_layer_numeric, mean), 
             y = upstroke_downstroke_ratio_rheo)) +
  
  # Geoms
  geom_boxplot(outlier.shape = NA, fill = "grey95") +
  geom_jitter(
    aes(color = vulnerability_group), 
    width = 0.2, 
    alpha = 0.9
  ) +
  
  # Scales, Labs, and Theme
  scale_color_manual(values = vulnerability_colors) +
  labs(
    y = "Upstroke/Downstroke ratio",
    x = "Supertype",
    color = "Vulnerability Group"
  ) + 
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position = "right"
  )

cortical_layers_plot / sag_plot / tau_plot 


# load L1 dataset - this is a mess, sorry
load('~/Github/patch_seq_lee/patchseq_human_L1-main/data/ps_human.RData')

l1_meta = read_csv('~/Github/patch_seq_lee/patchseq_human_L1-main/data/human_l1_dataset_2023_02_06.csv')
l1_meta = left_join(l1_meta, patch_seq_scanvi, join_by(exp_component_name == exp_component_name))


aibs_features_E = read_csv('~/Github/patch_seq_lee/patchseq_human_L1-main/data/aibs_features_E.csv')
mansvelder_features_E = read_csv('~/Github/patch_seq_lee/patchseq_human_L1-main/data/mansvelder_features_E.csv')
tamas_features_E = read_csv('~/Github/patch_seq_lee/patchseq_human_L1-main/data/tamas_features_E.csv')

aibs_features_E$cell_name = as.character(aibs_features_E$cell_name)
bind_rows(aibs_features_E, mansvelder_features_E)

aibs_l1_data = left_join(l1_meta, aibs_features_E)

aibs_features_E = right_join(annoPS, aibs_features_E, join_by(spec_id_label == cell_name))
mansvelder_features_E = right_join(annoPS, mansvelder_features_E, join_by(sample_id == cell_name))
tamas_features_E = right_join(annoPS, tamas_features_E, join_by(sample_id == cell_name))

l1_ephys_meta_all = bind_rows(bind_rows(aibs_features_E, mansvelder_features_E), tamas_features_E)

# this poorly combines the lee dalley dataset with the l1 dataset
combined_ephys_all = bind_rows(joined_dataset %>% dplyr::select(-Norm_Marker_Sum.0.4_label), l1_ephys_meta_all)

combined_ephys_all %>% ggplot(aes(x = supertype_scANVI, y = sag)) + 
  geom_boxplot() + 
  geom_point(alpha = 0.1) + 
  facet_wrap(~subclass_scANVI, scales = "free") + 
    theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
