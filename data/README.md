# Dataset

## Source

This project uses the **Maternal Health Risk Assessment Dataset** published by:

> Mojumdar, M. U., Assaduzzaman, M., Sarker, D., Shifa, H. A., Sajeeb, M. A. H., Bari, S., Chakraborty, N. R., & Alam, M. J. (2024). *Maternal Health Risk Assessment Dataset*. Mendeley Data, Version 1.

Dataset DOI:

**10.17632/p5w98dvbbk.1**

Dataset record:

https://data.mendeley.com/datasets/p5w98dvbbk/1

The published paper describes the dataset used in this study as containing **1,205 records and 12 columns**, consisting of 11 predictors and the `Risk Level` target.

## Variables used

The predictors used in the published study were:

- Age
- Systolic BP
- Diastolic BP
- BS
- Body Temp
- BMI
- Previous Complications
- Preexisting Diabetes
- Gestational Diabetes
- Mental Health
- Heart Rate

Target:

- Risk Level

The historical experiment treated missing `Risk Level` values as a separate `Unknown` class.

## Redistribution

The Mendeley dataset record currently lists the dataset under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

The raw dataset is **not redistributed in this repository**. Users should obtain the dataset directly from the original Mendeley Data record and comply with its current licensing and attribution requirements.

## Reproduction note

The preserved historical notebook expects a local/Colab file named:

`Final.csv`

The repository does not include that raw file. The notebook should therefore be treated as a record of the historical experiment rather than a zero-configuration executable script.

Before reproducing the experiment, download the dataset from the original source and prepare the file expected by the notebook.
