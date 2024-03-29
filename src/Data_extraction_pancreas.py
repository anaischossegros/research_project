import pandas as pd
from pathlib import Path
import numpy as np
import json 


#First, we get the gene expression from TCGA 
data_path = Path("C:/Users/anais/Documents/2-Imperial/0-Research-Project/4-NN/Cox-nnet model/data/mrna_2")
datas = []
datas_id = []
for fp in data_path.glob('*.tsv'):
    file_path = fp
    datas_id.append(fp.name)
    data = pd.read_csv(fp, sep="\t", names = ['gene_id',	'gene_name',	'gene_type',	'unstranded',	'stranded_first',	'stranded_second',	'tpm_unstranded',	'fpkm_unstranded',	'fpkm_uq_unstranded'], usecols = [3], skiprows = 6)
    datas.append(data)
x_df = pd.concat([df.unstranded for df in datas], axis=1)
x_df.columns = [f"case_{i}_copy_number" for i in range(len(datas))]

#we make a dictionnary to associate the clinical data id to the gene expression id
#pancreatic
metadata_path = Path("C:/Users/anais/Documents/2-Imperial/0-Research-Project/4-NN/Cox-nnet model/data/metadata.cart.2022-06-24.json")
f = open(metadata_path)
metadata = json.load(f)
transcriptome_clinical_id = dict()
for k in range (len(metadata)): 
    case = metadata[k]
    for c in case['associated_entities']: 
        transcriptome_clinical_id[c['entity_submitter_id'][0:12]]=[case['file_name']]

#we get the clinical data
output_path = Path('C:/Users/anais/Documents/2-Imperial/0-Research-Project/4-NN/Cox-nnet model/data/clinical.tsv')
output_df = pd.read_csv(output_path, sep="\t", names = ['case_id',	'case_submitter_id',	'project_id',	'age_at_index',	'age_is_obfuscated', 'cause_of_death', 'cause_of_death_source',	'country_of_residence_at_enrollment',	'days_to_birth',	'days_to_death', 'ethnicity',	'gender',	'occupation_duration_years',	'premature_at_birth',	'race',	'vital_status', 'weeks_gestation_at_birth',	'year_of_birth',	'year_of_death',	'adrenal_hormone', 'age_at_diagnosis',	'ajcc_clinical_m',	'ajcc_clinical_n',	'ajcc_clinical_stage',	'ajcc_clinical_t', 'ajcc_pathologic_m',	'ajcc_pathologic_n',	'ajcc_pathologic_stage',	'ajcc_pathologic_t', 'ajcc_staging_system_edition',	'anaplasia_present',	'anaplasia_present_type', 'ann_arbor_b_symptoms',	'ann_arbor_b_symptoms_described',	'ann_arbor_clinical_stage', 'ann_arbor_extranodal_involvement',	'ann_arbor_pathologic_stage',	'best_overall_response', 'breslow_thickness',	'burkitt_lymphoma_clinical_variant',	'child_pugh_classification', 'circumferential_resection_margin',	'classification_of_tumor',	'cog_liver_stage', 'cog_neuroblastoma_risk_group',	'cog_renal_stage',	'cog_rhabdomyosarcoma_risk_group', 'days_to_best_overall_response',	'days_to_diagnosis',	'days_to_last_follow_up', 'days_to_last_known_disease_status',	'days_to_recurrence',	'eln_risk_classification', 'enneking_msts_grade',	'enneking_msts_metastasis',	'enneking_msts_stage', 'enneking_msts_tumor_site',	'esophageal_columnar_dysplasia_degree', 'esophageal_columnar_metaplasia_present',	'figo_stage',	'figo_staging_edition_year', 'first_symptom_prior_to_diagnosis',	'gastric_esophageal_junction_involvement', 'gleason_grade_group',	'gleason_grade_tertiary',	'gleason_patterns_percent', 'goblet_cells_columnar_mucosa_present',	'greatest_tumor_dimension', 'gross_tumor_weight', 'icd_10_code',	'igcccg_stage',	'inpc_grade',	'inpc_histologic_group',	'inrg_stage',	'inss_stage', 'international_prognostic_index',	'irs_group',	'irs_stage',	'ishak_fibrosis_score',	'iss_stage', 'largest_extrapelvic_peritoneal_focus',	'last_known_disease_status',	'laterality', 'lymph_node_involved_site',	'lymph_nodes_positive',	'lymph_nodes_tested', 'lymphatic_invasion_present',	'margin_distance',	'margins_involved_site',	'masaoka_stage' 'medulloblastoma_molecular_classification',	'metastasis_at_diagnosis', 'metastasis_at_diagnosis_site',	'method_of_diagnosis',	'micropapillary_features', 'mitosis_karyorrhexis_index',	'mitotic_count',	'morphology',	'non_nodal_regional_disease', 'non_nodal_tumor_deposits',	'ovarian_specimen_status',	'ovarian_surface_involvement', 'papillary_renal_cell_type',	'percent_tumor_invasion',	'perineural_invasion_present', 'peripancreatic_lymph_nodes_positive',	'peripancreatic_lymph_nodes_tested', 'peritoneal_fluid_cytological_status',	'pregnant_at_diagnosis',	'primary_diagnosis', 'primary_disease', 'primary_gleason_grade',	'prior_malignancy',	'prior_treatment',	'progression_or_recurrence', 'residual_disease',	'satellite_nodule_present',	'secondary_gleason_grade',	'site_of_resection_or_biopsy',	'sites_of_involvement', 'supratentorial_localization',	'synchronous_malignancy',	'tissue_or_organ_of_origin', 'transglottic_extension',	'tumor_confined_to_organ_of_origin',	'tumor_depth',	'tumor_focality', 'tumor_grade',	'tumor_largest_dimension_diameter',	'tumor_regression_grade', 'tumor_stage', 'vascular_invasion_present',	'vascular_invasion_type',	'weiss_assessment_score', 'who_cns_grade', 'who_nte_grade',	'wilms_tumor_histologic_subtype',	'year_of_diagnosis', 'chemo_concurrent_to_radiation',	'days_to_treatment_end',	'days_to_treatment_start', 'initial_disease_status',	'number_of_cycles',	'reason_treatment_ended', 'regimen_or_line_of_therapy',	'route_of_administration',	'therapeutic_agents', 'treatment_anatomic_site',	'treatment_arm',	'treatment_dose',	'treatment_dose_units', 'treatment_effect',	'treatment_effect_indicator',	'treatment_frequency', 'treatment_intent_type',	'treatment_or_therapy',	'treatment_outcome',	'treatment_type', '?'], usecols = ['case_submitter_id', 'vital_status', 'days_to_death', 'days_to_last_follow_up','age_at_diagnosis'], skiprows = 4)

#we sort the clinical data and the gene expression data in the same order
file_name=[]
for k, i in enumerate(output_df['case_submitter_id']):
    if i in transcriptome_clinical_id.keys():
        file_name.append(transcriptome_clinical_id[i][0])
    else: 
        output_df = output_df.drop(k, axis = 0)
output_df = output_df.assign(filename=file_name)
output_df = output_df.sort_values(by=['filename', 'vital_status'], ascending = [True, True])
output_df = output_df.drop_duplicates(subset=["filename"], keep='last') #remove duplicate
output_df = output_df.drop(output_df[output_df.age_at_diagnosis =="'--"].index, axis=0) #remove data with no age
output_df = output_df.drop(output_df[(output_df.days_to_death ==output_df.days_to_last_follow_up) & (output_df.days_to_death=="'--")].index, axis=0) #remove data with neither death nor last follow up
output_df = output_df.drop(output_df[(output_df.days_to_last_follow_up == 'NaN') & (output_df.days_to_death=="'--")].index, axis=0) #remove data with neither death nor last follow up


#we remove the inputs which have no output
list= np.array(output_df['filename'])
missing=[]
for m in datas_id: 
    if m not in list: 
        missing.append(m)

x_df2 = x_df.transpose()
x_df2 = x_df2.assign(filename=datas_id)
x_df2= x_df2.set_index(['filename'])
x_df2 = x_df2.drop(missing, axis = 0)
output_df_pancreas= output_df.set_index(['filename'])

#export the input
x_df2.to_csv('x_df2_pancreas.csv')

#import the normalised input
data_norm_path = Path("C:/Users/anais/Documents/2-Imperial/0-Research-Project/4-NN/Cox-nnet model/data/normalized_pancreas.txt")
data_norm_df = pd.read_csv(data_norm_path, sep=" ")
data_norm_df_pancreas = data_norm_df.transpose()
