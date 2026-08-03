"""
railway_reference_data.py
--------------------------
Static reference data for the Indian Railways network used as the seed
topology for synthetic dataset generation.

This module intentionally hard-codes real Indian Railways zone codes and a
representative set of major/junction stations per zone (approx. lat/long).
It is NOT the full ~7,325 station network (per NFR-02 that is the scale
target for production, out of scope for a synthetic BE project dataset) —
instead it selects high-traffic junctions and division headquarters per
zone so that the resulting graph is topologically realistic (hub-and-spoke
around zonal HQs, inter-zone junctions with high betweenness centrality).

Author: RailwayCascadeAI Project (BE Major Project - Phase 2)
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ZoneInfo:
    zone_id: str
    zone_name: str
    headquarter: str


# 17 administrative zones of Indian Railways (zone_id = official 2-3 letter code)
ZONES: List[ZoneInfo] = [
    ZoneInfo("CR", "Central Railway", "CSMT"),
    ZoneInfo("ER", "Eastern Railway", "HWH"),
    ZoneInfo("ECR", "East Central Railway", "PNBE"),
    ZoneInfo("ECoR", "East Coast Railway", "BBS"),
    ZoneInfo("NR", "Northern Railway", "NDLS"),
    ZoneInfo("NCR", "North Central Railway", "PRYJ"),
    ZoneInfo("NER", "North Eastern Railway", "GKP"),
    ZoneInfo("NFR", "Northeast Frontier Railway", "GHY"),
    ZoneInfo("NWR", "North Western Railway", "JP"),
    ZoneInfo("SR", "Southern Railway", "MAS"),
    ZoneInfo("SCR", "South Central Railway", "SC"),
    ZoneInfo("SECR", "South East Central Railway", "BSP"),
    ZoneInfo("SER", "South Eastern Railway", "KGP"),
    ZoneInfo("SWR", "South Western Railway", "SBC"),
    ZoneInfo("WR", "Western Railway", "BCT"),
    ZoneInfo("WCR", "West Central Railway", "JBP"),
    ZoneInfo("KR", "Konkan Railway", "CDL"),
]

# Representative stations per zone.
# Fields: (station_code, station_name, zone_id, lat, lon, platform_count, traffic_category)
# traffic_category in {HIGH, MEDIUM, LOW}; HIGH stations are zonal HQs / major junctions.
STATIONS = [
    # Central Railway (CR)
    ("CSMT", "Chhatrapati Shivaji Maharaj Terminus", "CR", 18.9398, 72.8355, 18, "HIGH"),
    ("PUNE", "Pune Junction", "CR", 18.5286, 73.8744, 6, "HIGH"),
    ("NGP", "Nagpur Junction", "CR", 21.1523, 79.0882, 8, "HIGH"),
    ("BSL", "Bhusaval Junction", "CR", 21.0447, 75.7849, 7, "MEDIUM"),
    ("SUR", "Solapur Junction", "CR", 17.6599, 75.9064, 6, "MEDIUM"),
    ("KYN", "Kalyan Junction", "CR", 19.2437, 73.1355, 8, "HIGH"),
    ("DD", "Daund Junction", "CR", 18.4642, 74.5815, 5, "LOW"),

    # Eastern Railway (ER)
    ("HWH", "Howrah Junction", "ER", 22.5851, 88.3468, 23, "HIGH"),
    ("SDAH", "Sealdah", "ER", 22.5675, 88.3708, 20, "HIGH"),
    ("ASN", "Asansol Junction", "ER", 23.6739, 86.9524, 8, "MEDIUM"),
    ("BWN", "Barddhaman Junction", "ER", 23.2324, 87.8615, 6, "MEDIUM"),
    ("MLDT", "Malda Town", "ER", 25.0217, 88.1414, 5, "LOW"),

    # East Central Railway (ECR)
    ("PNBE", "Patna Junction", "ECR", 25.6093, 85.1376, 10, "HIGH"),
    ("DNR", "Danapur", "ECR", 25.6328, 85.0453, 5, "LOW"),
    ("MFP", "Muzaffarpur Junction", "ECR", 26.1225, 85.3906, 6, "MEDIUM"),
    ("GAYA", "Gaya Junction", "ECR", 24.7955, 85.0002, 7, "MEDIUM"),
    ("DHN", "Dhanbad Junction", "ECR", 23.7957, 86.4304, 7, "MEDIUM"),

    # East Coast Railway (ECoR)
    ("BBS", "Bhubaneswar", "ECoR", 20.2700, 85.8330, 6, "HIGH"),
    ("PURI", "Puri", "ECoR", 19.8106, 85.8314, 5, "MEDIUM"),
    ("CTC", "Cuttack Junction", "ECoR", 20.4670, 85.8830, 6, "MEDIUM"),
    ("VSKP", "Visakhapatnam", "ECoR", 17.7231, 83.3040, 8, "HIGH"),
    ("BAM", "Brahmapur", "ECoR", 19.3150, 84.7940, 4, "LOW"),

    # Northern Railway (NR)
    ("NDLS", "New Delhi", "NR", 28.6431, 77.2197, 16, "HIGH"),
    ("DLI", "Delhi Junction (Old Delhi)", "NR", 28.6606, 77.2274, 10, "HIGH"),
    ("UMB", "Ambala Cantt Junction", "NR", 30.3488, 76.8125, 8, "MEDIUM"),
    ("LDH", "Ludhiana Junction", "NR", 30.9010, 75.8573, 6, "MEDIUM"),
    ("ASR", "Amritsar Junction", "NR", 31.6340, 74.8723, 6, "MEDIUM"),
    ("SVDK", "Shri Mata Vaishno Devi Katra", "NR", 33.0308, 74.5335, 4, "LOW"),

    # North Central Railway (NCR)
    ("PRYJ", "Prayagraj Junction", "NCR", 25.4484, 81.8546, 10, "HIGH"),
    ("AGC", "Agra Cantt", "NCR", 27.1560, 77.9910, 6, "MEDIUM"),
    ("JHS", "Jhansi Junction", "NCR", 25.4484, 78.5685, 7, "MEDIUM"),

    # North Eastern Railway (NER)
    ("GKP", "Gorakhpur Junction", "NER", 26.7606, 83.3732, 10, "HIGH"),
    ("BST", "Basti", "NER", 26.8145, 82.7645, 4, "LOW"),
    ("GD", "Gonda Junction", "NER", 27.1333, 81.9667, 5, "LOW"),

    # Northeast Frontier Railway (NFR)
    ("GHY", "Guwahati", "NFR", 26.1839, 91.7362, 6, "HIGH"),
    ("NJP", "New Jalpaiguri", "NFR", 26.6961, 88.4310, 6, "MEDIUM"),
    ("DBRG", "Dibrugarh", "NFR", 27.4728, 94.9120, 4, "LOW"),

    # North Western Railway (NWR)
    ("JP", "Jaipur Junction", "NWR", 26.9196, 75.7878, 7, "HIGH"),
    ("JU", "Jodhpur Junction", "NWR", 26.2870, 73.0243, 6, "MEDIUM"),
    ("BKN", "Bikaner Junction", "NWR", 28.0143, 73.3119, 5, "LOW"),
    ("AII", "Ajmer Junction", "NWR", 26.4570, 74.6390, 6, "MEDIUM"),

    # Southern Railway (SR)
    ("MAS", "Chennai Central", "SR", 13.0827, 80.2755, 12, "HIGH"),
    ("MS", "Chennai Egmore", "SR", 13.0732, 80.2609, 10, "MEDIUM"),
    ("TPJ", "Tiruchchirapalli Junction", "SR", 10.8038, 78.6857, 7, "MEDIUM"),
    ("MDU", "Madurai Junction", "SR", 9.9252, 78.1198, 6, "MEDIUM"),
    ("CBE", "Coimbatore Junction", "SR", 11.0018, 76.9629, 6, "MEDIUM"),
    ("TVC", "Thiruvananthapuram Central", "SR", 8.4875, 76.9525, 5, "HIGH"),
    ("ERS", "Ernakulam Junction", "SR", 9.9714, 76.2842, 5, "MEDIUM"),

    # South Central Railway (SCR)
    ("SC", "Secunderabad Junction", "SCR", 17.4344, 78.5008, 10, "HIGH"),
    ("HYB", "Hyderabad Deccan", "SCR", 17.3819, 78.4867, 6, "MEDIUM"),
    ("BZA", "Vijayawada Junction", "SCR", 16.5175, 80.6167, 10, "HIGH"),
    ("GNT", "Guntur Junction", "SCR", 16.3067, 80.4365, 5, "LOW"),
    ("NLDA", "Nalgonda", "SCR", 17.0575, 79.2670, 3, "LOW"),

    # South East Central Railway (SECR)
    ("BSP", "Bilaspur Junction", "SECR", 22.0797, 82.1391, 8, "HIGH"),
    ("R", "Raipur Junction", "SECR", 21.2379, 81.6337, 7, "MEDIUM"),
    ("DURG", "Durg Junction", "SECR", 21.1904, 81.2849, 6, "MEDIUM"),

    # South Eastern Railway (SER)
    ("KGP", "Kharagpur Junction", "SER", 22.3302, 87.3237, 9, "HIGH"),
    ("TATA", "Tatanagar Junction", "SER", 22.7925, 86.1842, 6, "MEDIUM"),
    ("ROU", "Rourkela", "SER", 22.2604, 84.8536, 5, "LOW"),

    # South Western Railway (SWR)
    ("SBC", "KSR Bengaluru City Junction", "SWR", 12.9767, 77.5713, 10, "HIGH"),
    ("UBL", "Hubballi Junction", "SWR", 15.3487, 75.1240, 6, "MEDIUM"),
    ("MYS", "Mysuru Junction", "SWR", 12.3072, 76.6553, 5, "MEDIUM"),

    # Western Railway (WR)
    ("BCT", "Mumbai Central", "WR", 18.9694, 72.8202, 7, "HIGH"),
    ("ADI", "Ahmedabad Junction", "WR", 23.0225, 72.5714, 12, "HIGH"),
    ("ST", "Surat", "WR", 21.1959, 72.8302, 5, "MEDIUM"),
    ("RTM", "Ratlam Junction", "WR", 23.3315, 75.0367, 6, "MEDIUM"),
    ("BVP", "Bhavnagar Terminus", "WR", 21.7645, 72.1519, 4, "LOW"),

    # West Central Railway (WCR)
    ("JBP", "Jabalpur Junction", "WCR", 23.1687, 79.9333, 7, "HIGH"),
    ("BPL", "Bhopal Junction", "WCR", 23.2680, 77.4020, 6, "HIGH"),
    ("COR", "Kota Junction", "WCR", 25.1804, 75.8648, 8, "MEDIUM"),

    # Konkan Railway (KR)
    ("CDL", "Kudal", "KR", 16.0092, 73.6800, 3, "LOW"),
    ("MAO", "Madgaon Junction", "KR", 15.3781, 73.9564, 5, "MEDIUM"),
    ("RN", "Ratnagiri", "KR", 16.9902, 73.3120, 4, "LOW"),
]

# Inter-zone junction corridors: pairs of (station_code_A, station_code_B) that
# straddle a zonal boundary and are frequently used as cross-zone through-routes.
# These are used to explicitly seed a subset of edges with the zone_crossing flag.
CROSS_ZONE_CORRIDORS = [
    ("BSP", "R"), ("R", "DURG"), ("NGP", "BSP"), ("NGP", "R"),
    ("BZA", "GNT"), ("BZA", "VSKP"), ("VSKP", "BBS"),
    ("SC", "SBC"), ("SC", "BZA"), ("SBC", "UBL"),
    ("UBL", "MAO"), ("MAO", "CDL"), ("KYN", "MAO"),
    ("PRYJ", "PNBE"), ("PNBE", "GAYA"), ("GAYA", "DHN"), ("DHN", "ASN"),
    ("KGP", "TATA"), ("TATA", "ROU"), ("ROU", "BSP"),
    ("NDLS", "PRYJ"), ("PRYJ", "JHS"), ("JHS", "BPL"), ("BPL", "NGP"),
    ("BPL", "JBP"), ("JBP", "BSP"),
    ("RTM", "COR"), ("COR", "NDLS"), ("ADI", "RTM"),
    ("BCT", "ST"), ("ST", "ADI"),
    ("NDLS", "UMB"), ("UMB", "ASR"), ("UMB", "LDH"),
    ("JP", "COR"), ("JP", "AII"), ("AII", "JU"), ("JU", "BKN"),
    ("GKP", "MFP"), ("MFP", "PNBE"), ("GD", "GKP"), ("BST", "GKP"),
    ("GHY", "NJP"), ("NJP", "MLDT"), ("MLDT", "ASN"),
    ("MAS", "SC"), ("MAS", "TPJ"), ("TPJ", "MDU"), ("MDU", "TVC"),
    ("CBE", "ERS"), ("ERS", "TVC"), ("TPJ", "CBE"),
    ("HWH", "ASN"), ("HWH", "KGP"), ("SDAH", "MLDT"),
]
