# **SAFEVAR**

## **Description**
Autonomous driving systems (ADSs) must be sufficiently tested to ensure their safety. Though various ADS testing methods have shown promising results, they are limited to a fixed set of vehicle characteristics settings (VCSs). The impact of variations in vehicle characteristics (e.g., mass, tire friction) on the safety of ADSs has not been sufficiently and systematically studied. Such variations are often due to wear and tear, production errors, etc., which may lead to unexpected driving behaviours of ADSs. To this end, in this paper, we propose a method, named SafeVar, to systematically find minimum variations to the original vehicle characteristics setting, which affect the safety of the ADS deployed on the vehicle. To evaluate the effectiveness of SafeVar, we employed an ADS and conducted experiments with two driving scenarios. Results show that SafeVar, equipped with NSGA-II, generates more critical VCSs that put the vehicle into unsafe situations, as compared with the baseline algorithm: Random Search. We also identified critical vehicle characteristics and reported to which extent varying their settings put the ADS vehicle into unsafe situations

This repository contains:

1. **DataSet** : all the raw data for the analyses (including two settings);
2. **Source code** of the scenario designed in the CARLA simulator and the code of the combination of the extended WOR and NSGA-II algorithm (RS); The "**WorldOnRails2.0**" folder contains relevant experiments and documents for Carla Sun (Carla Rain) experiment;
4. **Supplement**: We have provided supplementary data for RQ3.3 (in the paper) in the "Supplement" folder;



## **Contributions**
We extended this work from the following aspects:
- We proposed SAFEVAR to generate variations to vehicle characteristics that threaten the safety of the ADS (not the Automatic Emergency Brake operation); 
- We designed more realistic driving scenarios (e.g., pedestrians crossing the road); 
- With the CARLA simulator, we conducted experiments considering two different weather conditions; 
- To more comprehensively evaluate SAFEVAR, we introduced more safety metrics in addition to the safety metric safetyDegree that we use to drive the search (i.e., Time Exposed Time-to-collision (TET), Time Integrated Time-to-collision (TIT), and average deceleration (aveDece)).

## **Prerequisite**
- [CARLA 0.9.10](https://carla.readthedocs.io/en/0.9.10/)  
- [jMetalPy](https://github.com/jMetal/jMetalPy)
- **Python** :for CARLA, you should refer to the [World On Rail](https://github.com/dotchen/WorldOnRails/blob/release/docs/INSTALL.md) to set up environment and the [WordOnRail2.0 README]() to run program; for LGSVL, you should refer to Apollo and using python 3.7;

## People
- Qi Pan
- Paolo Arcaini http://group-mmm.org/~arcaini/
- Tao Yue https://www.simula.no/people/tao
- Shaukat Ali https://www.simula.no/people/shaukat
