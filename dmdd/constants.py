YEAR_IN_S = 31557600.
GEV_IN_KEV = 1.e6
C_KMSEC = 299792.458

NUCLEON_MASS = 0.938272 # Nucleon mass in GeV
P_MAGMOM = 2.793 # proton magnetic moment, PDG Live
N_MAGMOM = -1.913 # neutron magnetic moment, PDG Live

NUCLEAR_MASSES = {
    'xenon': 122.298654871,
    'germanium': 67.663731424,
    'argon': 37.2113263068,
    'silicon': 26.1614775455,
    'sodium': 21.4140502327,
    'iodine': 118.206437626,
    'fluorine': 17.6969003039,
    } # this is target nucleus mass in GeV: mT[GeV] = 0.9314941 * A[AMU]

ELEMENT_INFO = {"xenon":{128:0.0192,129:0.2644,130:0.0408,131:0.2118,132:0.2689,134:0.1044,136:0.0887,'weight':131.1626},"germanium":{70:0.2084,72:0.2754,73:0.0773,74:0.3628,76:0.0761,'weight':72.6905},"iodine":{127:1.,'weight':127.},"sodium":{23:1.,'weight':23.},"silicon":{28:0.922,29:0.047,30:0.031,'weight':28.109},"fluorine":{19:1.,'weight':19.},"argon":{40:1.,'weight':40.},"helium":{4:1.,'weight':4.}} 
    

