# **SAFEVAR**

## **Description**
Autonomous driving systems (ADSs) must be sufficiently tested to ensure their safety. Though various ADS testing methods have shown promising results, they are limited to a fixed set of vehicle characteristics settings (VCSs). The impact of variations in vehicle characteristics (e.g., mass, tire friction) on the safety of ADSs has not been sufficiently and systematically studied. Such variations are often due to wear and tear, production errors, etc., which may lead to unexpected driving behaviours of ADSs. T o this end, in this paper, we propose a method, named SAFEVAR, to systematically find minimum variations to the original vehicle characteristics setting, which affect the safety of the ADS deployed on the vehicle. T o evaluate the effectiveness of SAFEVAR, we employed two ADSs and conducted experiments with two driving simulators. Results show that SAFEVAR, equipped with NSGA-II, generates critical VCSs that put the vehicle into unsafe situations. Furthermore, we studied the impact of weather conditions. Experiment results show slight variations to some characteristics under certain weather conditions (e.g., rain) might put ADSs into unsafe situations such as collisions

This repository contains:

1. **DataSet** : all the raw data for the analyses (including three settings);
2. **Source code** of the scenario designed in the CARLA simulator and the code of the combination of the extended WOR and NSGA-II algorithm;
3. **Source code** of scenario designed in LGSVL combined with the NSGA-II algorithm and the LGSVL simulator with our own APIs deployed;

## **Contributions**
We extended this work from the following aspects:
- We proposed SAFEVAR to generate variations to vehicle characteristics that threaten the safety of ADSs (not the Automatic Emergency Brake operation); 
- We designed more realistic driving scenarios (e.g., pedestrians crossing the road, avoiding stationary vehicles ahead); 
- With the CARLA simulator, we conducted experiments considering two different weather conditions; 
- We experimented SAFEVAR with two ADSs (World On Rails and Apollo) and two simulators; 
- To more comprehensively evaluate SAFEVAR, we introduced more safety metrics in addition to the safety metric safetyDegree that we use to drive the search (i.e., Time Exposed Time-to-collision (TET), Time Integrated Time-to-collision (TIT), and average deceleration (aveDece)).

## **Prerequisite**
- [CARLA 0.9.10](https://carla.readthedocs.io/en/0.9.10/)  
- [Apollo 5.0](https://github.com/ApolloAuto/apollo/tree/v5.0.0)
- [jMetalPy](https://github.com/jMetal/jMetalPy)
- **Python** :for CARLA, you should refer to the [World On Rail](https://github.com/dotchen/WorldOnRails/blob/release/docs/INSTALL.md) to set up environment and the [WordOnRail2.0 Readme]() to run program; for LGSVL, you should refer to Apollo and using python 3.7;

## People
- Qi Pan
- Paolo Arcaini http://group-mmm.org/~arcaini/
- Tao Yue https://www.simula.no/people/tao
- Shaukat Ali https://www.simula.no/people/shaukat