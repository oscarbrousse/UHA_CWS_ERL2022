# Urban Heat Advection using Citizen Weather Stations (ERL paper: Brousse et al. 2022)
Repository of all scripts used in the ERL paper focusing on urban heat advection using Netatmo CWS, entitled: "Evidence of horizontal urban heat advection in London using 6 years of data from a citizen weather station network"

### Workflow:
1. **Netatmo data acquisition**

Any user who would like to download Netatmo data can do it using their API. Here, we first downloaded a list of all stations available in the domain of interest by running the Collect_NetAtmo_Data_Tile_ERL.py script that uses the _Getpublicdata_ API. This script has a moving window of 0.2 ° in both directions (latitude and longitude) that ensures the download of all possible stations in the domain. Do not increase the size of this moving window. Once the list of all stations in the domain is created, we run the Collect_NetAtmo_Data_ERL.py script that uses the _Getmeasure_ API. In this study, we only download air temperature data but other data can be downloaded (see the Netatmo API documentation: https://nobodyinperson.gitlab.io/python3-patatmo/whatsthis.html). In Collect_NetAtmo_Data_ERL.py, a testing script can be run to test the connection to the Netatmo client.

2. **MIDAS data acquisition**

The official weather stations MIDAS network from the UK MetOffice can be accessed on the CEDA website: https://catalogue.ceda.ac.uk/uuid/3bd7221d4844435dad2fa030f26ab5fd . We used the v202107 version. For other official automatic weather stations we do not provide support in the data treatment.

3. **Data treatment and filtering**

**Netatmo**
Once all the Netatmo data is acquired, you need to run a Python script before runing the filtering CrowdQC algorithm: Aggregate_NetAtmo_Measurements_ERL.py. Once all the data is prepared for the CrowdQC ingestion, run the Filter_CWS_ERL.R script. Then, use the Netatmo_Data_Management_ERL.py script to prepare the data for the UHA calculation, analysis and visualisation.

**MIDAS**
The MIDAS CSV data comes in a specific structure. We standardize the data structure for the UHA analysis using the Standardize_MIDAS_AWS_ERL.py and the MIDAS_DataTreatment_and_WindRoses_ERL.py scripts. The latter script also produces the windroses graphs plotted in the ERL manuscript's supplementary materials

4. **Data treatment and visualisation**
For visualising where the CWS data are located over a Local Climate Zones map, one has to use the DataLocation_LCZMap_Domains_ERL.py script. For calculating the urban heat advection (UHA) and obtaining all the results presented in the manuscript concerning the urban temperatures and the daily temperature ranges measured by CWS, the NetAtmo_UHI_per_MIDAS_Wind_ERL.py script has to be used. In this script, MIDAS wind orientations and speeds are classified into prevailing wind categories (see the manuscript's methods for more info) that serve the calculation of the UHA.

### Additional tools:

The MIDAS_Wind_Comparison.py script can be run to compare the official AWS against others in the urban environment to see how often the prevailing wind characterized at the chose AWS is consistent with other surrounding stations.
