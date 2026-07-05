# Rhizosheath_MaxEnt_SDM_2026
Code repository for global rhizosheath species distribution modelling using MaxEnt, including raster preprocessing, predictor screening, future projections and figure generation.

# Rhizosheath_SDM_Code

Code repository for global rhizosheath species distribution modelling using MaxEnt, including raster preprocessing, predictor screening, MaxEnt input preparation, post-processing, future climate projections and figure generation.

## Repository status

This repository accompanies the manuscript:

**[Manuscript title to be inserted]**

A citable archived version of this repository will be deposited in Zenodo upon manuscript submission or acceptance.

**Repository DOI:** [Zenodo DOI to be inserted]

## Overview

This repository contains Python scripts used to prepare environmental predictors, generate MaxEnt input files, post-process MaxEnt outputs, summarize present-day and future suitability patterns, quantify projected changes in suitable areas and generate figures for the rhizosheath species distribution modelling analyses.

MaxEnt model fitting was conducted externally using **MaxEnt v3.4.4**. The scripts in this repository do not reimplement the MaxEnt algorithm. Instead, they document and reproduce the Python-based workflow used for:

1. Environmental raster preprocessing
2. Raster alignment and format conversion
3. Predictor value extraction
4. Predictor screening
5. MaxEnt input preparation
6. MaxEnt output post-processing
7. Future climate projection processing
8. Multi-GCM ensemble summarization
9. Suitability thresholding and change classification
10. Figure generation

## Repository structure

```text
Rhizosheath_SDM_Code/
│
├── README.md
├── requirements.txt
├── config_template.yaml
├── CITATION.cff
├── LICENSE
│
├── docs/
│   ├── MaxEnt_original_README.txt
│   ├── MaxEnt_parameter_settings.md
│   └── Supplementary_workflow_note.md
│
├── 01_data_preprocessing/
│   ├── Step01_BioclimCalculator.py
│   ├── Step02_RasterAlignmentCheck.py
│   ├── Step03_GLCFCS_MosaicAndResample.py
│   ├── Step04_HWSD2_SoilExtraction.py
│   └── Step05_TIFtoASC.py
│
├── 02_predictor_screening/
│   ├── Step01_ExtractEnvValues.py
│   ├── Step02_PearsonCorrelation.py
│   ├── Step03_CheckNodata.py
│   ├── Step04_PredictorSelectionSummary.py
│   └── Step05_FinalPredictorTable.py
│
├── 03_maxent_inputs/
│   ├── Step01_CreateOccurrenceCSV.py
│   ├── Step02_CreateMaxEntBatchFiles.py
│   └── Step03_CheckASCHeaders.py
│
├── 04_model_postprocessing/
│   ├── Step01_ReadMaxEntResults.py
│   ├── Step02_SuitabilityThresholding.py
│   ├── Step03_CurrentSuitabilityArea.py
│   └── Step04_ResponseCurveSummary.py
│
├── 05_future_projection/
│   ├── Step01_FutureClimateASCPreparation.py
│   ├── Step02_GCMProjectionStack.py
│   ├── Step03_EnsembleMedianAndUncertainty.py
│   ├── Step04_ChangeClassification.py
│   └── Step05_SSPAreaSummary.py
│
└── 06_figures/
    ├── Step01_CurrentMap.py
    ├── Step02_FutureChangeMap.py
    ├── Step03_UncertaintyMap.py
    └── Step04_SupplementaryDiagnostics.py
```

## Software requirements

The workflow was developed using:

* Python 3.10
* MaxEnt v3.4.4
* rasterio
* numpy
* pandas
* geopandas
* scipy
* scikit-learn
* matplotlib
* PyYAML

Additional package versions are listed in `requirements.txt`.

## Installation

Create a clean Python environment and install the required dependencies:

```bash
conda create -n rhizosheath_sdm python=3.10
conda activate rhizosheath_sdm
pip install -r requirements.txt
```

Alternatively, dependencies can be installed directly with:

```bash
pip install -r requirements.txt
```

## Configuration

The file `config_template.yaml` provides an example configuration file for local input and output paths.

Before running the scripts, copy the template and edit the paths according to the local file system:

```bash
cp config_template.yaml config.yaml
```

The local `config.yaml` file is intentionally excluded from version control because it contains machine-specific paths.

## Occurrence records

Occurrence records were provided as a CSV file containing three columns:

```text
Species, latitude, longitude
```

Duplicate records were removed before modelling. Spatial thinning was not applied because the final number of occurrence records was limited.

The occurrence table used for MaxEnt input preparation should follow the format required by MaxEnt, with species name and geographic coordinates in decimal degrees.

## Environmental predictors

### Present-day predictors

Present-day environmental predictors included climate, soil and land-cover variables.

Climate predictors were derived from monthly precipitation, minimum temperature and maximum temperature layers for the period **2000–2021**. For each year, 19 bioclimatic variables were calculated from monthly climate data. The annual bioclimatic layers were then averaged pixel by pixel to generate the final present-day climate predictors.

Soil predictors were derived from **HWSD2**. The upper 0–60 cm soil profile was represented using soil layers D1–D3. Soil variables were extracted and aligned to the common global grid.

Land-cover information was derived from **GLC_FCS10**, a 10 m global land-cover product with fine land-cover classification.

All environmental predictors were aligned to a common global grid at approximately 1 km spatial resolution. Continuous variables were resampled using bilinear interpolation, whereas categorical variables, including land-cover type and soil type identifiers, were resampled using the modal class.

### Final full-environment predictor set

Candidate predictors were screened based on pairwise Pearson correlation, data completeness and ecological interpretability. The final full-environment predictor set included eight climate variables, six soil variables and one land-cover variable.

The selected climate variables were:

```text
BIO1   Annual mean temperature
BIO4   Temperature seasonality
BIO7   Temperature annual range
BIO11  Mean temperature of coldest quarter
BIO12  Annual precipitation
BIO15  Precipitation seasonality
BIO16  Precipitation of wettest quarter
BIO17  Precipitation of driest quarter
```

The selected soil variables were:

```text
Soil pH
Soil organic carbon
Soil cation exchange capacity
Available water capacity
Clay content
Total nitrogen
```

The selected vegetation predictor was:

```text
GLC_FCS10 land-cover class
```

## Predictor screening

Predictor screening was conducted using the occurrence-point environmental values and global raster diagnostics.

The screening workflow included:

1. Extracting environmental values at occurrence locations
2. Calculating pairwise Pearson correlation coefficients
3. Checking raster completeness and NODATA proportions
4. Removing or avoiding highly redundant variables
5. Retaining variables with strong ecological interpretability
6. Preparing the final predictor table for MaxEnt

Highly correlated variables were not interpreted independently. When multiple predictors represented similar ecological dimensions, final selection prioritized ecological relevance, data completeness and interpretability.

VIF analysis was considered but was not used as the final screening criterion. Instead, predictor selection was based primarily on correlation structure, NODATA diagnostics and ecological reasoning.

## MaxEnt modelling

MaxEnt model fitting was conducted externally using **MaxEnt v3.4.4**.

The Python scripts in this repository prepare MaxEnt input files and post-process model outputs. The MaxEnt software itself should be run separately using the environmental layers and occurrence files generated by the scripts.

### Present-day full-environment model

The present-day full-environment model used climate, soil and land-cover predictors.

The final model settings were:

```text
Output format: cloglog
Replicate type: bootstrap
Number of replicates: 20
Random seed: enabled
Test percentage: 15
Background points: 15,000
Regularization multiplier: 1.5
Clamping: enabled
Fade by clamping: enabled
Response curves: enabled
Output grids: enabled
```

The full-environment model was used to evaluate the relative importance of climate, soil and land-cover predictors in explaining the present-day global distribution of rhizosheath occurrence.

### Climate-only model for future projection

Future projections used a climate-only MaxEnt model because equivalent future soil and land-cover predictors were not available.

The climate-only model used the selected climate predictors:

```text
BIO1, BIO4, BIO7, BIO11, BIO12, BIO15, BIO16, BIO17
```

The final climate-only model settings were:

```text
Output format: cloglog
Replicate type: bootstrap
Number of replicates: 10
Random seed: enabled
Test percentage: 15
Background points: 15,000
Regularization multiplier: 1.5
Clamping: enabled
Fade by clamping: enabled
Response curves: enabled
Output grids: enabled
```

## Future climate projections

Future projections were conducted for the mid-21st century period **2041–2060**.

The climate-only MaxEnt model was projected under three SSP scenarios:

```text
SSP126
SSP245
SSP585
```

Future climate projections used three CMIP6 GCMs:

```text
MRI-ESM2-0
MPI-ESM1-2-HR
EC-Earth3-Veg
```

For each SSP scenario, the model was projected to each of the three GCMs, resulting in nine future suitability rasters:

```text
3 GCMs × 3 SSP scenarios = 9 future projection rasters
```

Each projected raster represents cloglog suitability values ranging from 0 to 1.

## Multi-GCM ensemble processing

For each SSP scenario, the three GCM-specific suitability rasters were summarized pixel by pixel.

The primary ensemble statistic was the multi-GCM median suitability:

```text
Suitability_median = median(MRI-ESM2-0, MPI-ESM1-2-HR, EC-Earth3-Veg)
```

The multi-GCM median was used because it is less sensitive to extreme predictions from a single GCM than a simple arithmetic mean.

Projection uncertainty was quantified using the spread among GCM predictions. Two optional uncertainty metrics were calculated:

```text
Suitability_range = max(GCM suitability) - min(GCM suitability)
Suitability_sd    = standard deviation among GCM suitability values
```

Higher spread indicates greater disagreement among GCMs, whereas values close to zero indicate higher agreement among the three future climate projections.

## Suitability thresholding

Binary suitability maps were generated using the **10 percentile training presence cloglog threshold** from MaxEnt.

The final threshold used in this analysis was:

```text
P10 = 0.232
```

For both current and future predictions, pixels were classified as suitable or unsuitable using the same threshold:

```text
Suitability >= 0.232  → suitable
Suitability <  0.232  → unsuitable
```

Using the same threshold for current and future predictions ensures consistent comparison of present-day and projected suitability.

## Change classification

Projected changes in suitable areas were quantified by comparing binary current suitability with binary future suitability.

For the main change map, the SSP245 multi-GCM median projection was used.

Each pixel was assigned to one of four classes:

```text
Current = 0, Future = 0  →  Consistently unsuitable
Current = 1, Future = 0  →  Lost
Current = 0, Future = 1  →  Gained
Current = 1, Future = 1  →  Stable
```

The resulting four-class raster was used to map projected changes in global rhizosheath suitability under SSP245.

The same classification logic was also applied to SSP126, SSP245 and SSP585 for area statistics.

## Area statistics

Area statistics were calculated for each SSP scenario.

For each scenario, the workflow calculated:

```text
n0: consistently unsuitable pixels
n1: lost pixels
n2: gained pixels
n3: stable pixels
```

These pixel counts were then converted into area summaries and proportional summaries.


This table can be used to generate bar plots or stacked bar plots showing the relative proportions of lost, gained and stable suitable areas across SSP scenarios.

## Figure generation

The scripts in `06_figures/` generate manuscript and supplementary figures, including:

1. Present-day suitability map
2. Future change map under SSP245
3. Multi-GCM uncertainty map
4. SSP-specific area summary plots
5. Supplementary diagnostic figures

The main SSP245 change map uses the following categories:

text
Consistently unsuitable
Lost
Gained
Stable

The uncertainty map represents model spread among the three GCM projections.


## Reproducibility notes

The modelling extent was global. Environmental predictors were aligned to a common global grid. No additional regional mask or bias file was used in the final workflow.

Large raster datasets and third-party environmental products are not stored directly in this repository. Users should obtain these data from the original providers and organize them according to the directory structure specified in `config_template.yaml`.

Because MaxEnt was run externally, complete reproduction requires:

1. The Python scripts in this repository
2. The same input occurrence records
3. The same environmental predictor rasters
4. MaxEnt v3.4.4
5. The MaxEnt parameter settings documented above
6. The same post-processing scripts and thresholding rules

The original MaxEnt README is included in `docs/MaxEnt_original_README.txt` for software reference only. Project-specific workflow details are documented in this README and in the associated scripts.

## Interpretation notes

The suitability maps represent modelled environmental suitability based on the occurrence data and predictors used in this study. Areas with low predicted suitability should not be interpreted as confirmed absence or ecological impossibility.

This distinction is important because regions with limited occurrence records or uneven sampling effort may receive low predicted suitability due to sampling deficiency rather than true environmental unsuitability.

Future projections should be interpreted as potential changes in climate suitability rather than realized future distribution. The projections do not explicitly account for dispersal limitation, local adaptation, species interactions, future land-cover change or future soil change.

For the future projections, soil and land-cover predictors were not included because equivalent future layers were not available. Their effects are therefore not projected dynamically into the future.

## Citation

If you use or adapt the code in this repository, please cite the archived repository release:

```text
[Zenodo citation to be inserted]
```

Please also cite the associated manuscript:

```text
[Manuscript citation to be inserted after publication]
```

## License

Code in this repository is released under:

```text
[License to be inserted, e.g. MIT License]
```

Third-party datasets remain subject to their original licenses and terms of use.

## Contact

For questions about the workflow, please contact:

text
Jingkang Liao
Lanzhou University, China
Jingk.liao@gamil.com
```
