# -*- coding: utf-8 -*-
"""Popular datasets of the MIT-BIH Arrhythmia Database.

Notes
----
I'm using the popular inter-patient scheme division proposed by Chazal in his
paper: Automatic classification of heartbeats using ECG morphology and
heartbeat interval features. doi:https://doi.org/10.1109/TBME.2004.827359.
"""

CHAZAL_TRAIN_DATASET = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230',
]
"""list: The Chazal dataset for training.  
"""

CHAZAL_TEST_DATASET = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234',
]
"""list: The Chazal dataset for testing.  
"""

# Only patients that contain both MLII and V1
CHAZAL_REDUCED_TRAIN_DATASET = [
    '101', '106', '108', '109', '112', '115', '118', '119', '201', '203',
    '205', '207', '208', '209', '215', '220', '223', '230',
]
"""list: The Chazal dataset for training (only with patients that contain MLII and V1).  
"""

CHAZAL_REDUCED_TEST_DATASET = [
    '105', '111', '113', '121', '200', '202', '210', '212', '213', '214',
    '219', '221', '222', '228', '231', '232', '233', '234',
]
"""list: The Chazal dataset for testing (only with patients that contain MLII and V1).  
"""

MIT_BIH_ARRHYTHMIA_DB = [
    '100', '104', '108', '113', '117', '122', '201', '207', '212', '217',
    '222', '231', '101', '105', '109', '114', '118', '123', '202', '208',
    '213', '219', '223', '232', '102', '106', '111', '115', '119', '124',
    '203', '209', '214', '220', '228', '233', '103', '107', '112', '116',
    '121', '200', '205', '210', '215', '221', '230', '234',
]
"""list: All the existing records in the MIT-BIH Arrhythmia Database.
"""
