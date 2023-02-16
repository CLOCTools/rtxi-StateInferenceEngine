#include "lfpInferenceEngine.h"

#include <chrono>

using namespace std;

// defining what's in the object's constructor
// user defines time window length (in samples) and sampling rate
lfpInferenceEngine::lfpInferenceEngine() {

    int i;

    setenv("PYTHONPATH","./pymodules:~/anaconda3/envs/hmm/lib/python3.10/site-packages",1);
    std::cout << "Initializing python environment.\n";
    Py_Initialize();
	PyRun_SimpleString("import sys, os");
    PyRun_SimpleString("print(sys.path)");
	//PyRun_SimpleString("sys.path.append(\"C:/Users/dweiss38/Documents/GitHub/utils-CorticalState/python\")");
	PyRun_SimpleString("sys.path.append(os.path.dirname(os.getcwd()))");
    PyRun_SimpleString("print(sys.path)");
    /*FILE* fp = fopen("./pymodules/pyfuncs.py", "r");
    int re = PyRun_SimpleFile(fp, "./pymodules/pyfuncs.py");
    */
    PyRun_SimpleString("print('Python Session started.')");

}

// defining what's in the object's destructor
lfpInferenceEngine::~lfpInferenceEngine(void) {
    std::cout << "Ending Python session.\n";
    Py_DECREF(pResult);
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    std::cout << "Python session ended. \n";
}

void lfpInferenceEngine::init(string recording, string modelname) {

    std::cout << "Loading model and initializing... \n";
    std::vector<std::string> arguments_load = {"pyfuncs","get_model", recording, modelname};
    this->callPythonFunction(arguments_load,{});
    this->load();
    
    //std::vector<std::string> arguments_loaddata = {"pyfuncs","get_data", recording, modelname};
    //this->callPythonFunction(arguments_loaddata,{});
    //this->load_data();
    N = 20;
    fftdata = { {1969.721422253957,3341.84545560465,6869.844489746644,7234.06379356454,6153.501956693461,5368.769008584881,2825.2020693233435,1203.5962894173047,1552.9074277122888,2078.558935106937,2122.9398361109734,524.3206215536861,387.7698207524475,803.3900212514357,977.5228585300412,1960.9667595849273,597.3258607908215,655.2721440603931,1596.8940064700053,1744.5044707714035,936.2739324748235,948.067910989219,636.0422586152944,345.91199199749377,306.85620535543666,534.7585339237452,542.5686008895664,596.9000629594669,157.900527453892,557.6249801294633,446.9888510755495,626.3924283681133,164.76931022696243,180.79887917998315,175.28848236358786,266.96042233404813,188.64667673365526,210.13340229721672,351.6572493964728,437.5663072819595,805.2307568222233,745.3487295865713,620.4089164987195,500.5799872852612,519.1975088341065,399.37683946152293,691.9481244074034,933.4649119146372,603.5798870535829,667.367143375828,851.8658642944087},
                {2670.1098995455495,3056.885004460356,6732.351867644141,6328.782994497914,6252.22464726597,5402.4279125015955,3016.6301119862687,817.9521218718556,1222.7493006921284,1745.1231968672955,2064.3629065847012,896.2608456688398,680.645238852832,705.1063179058664,898.2201262354489,2055.943831186252,552.5925907794383,653.9965221596839,1326.5296723216422,1639.6942613287115,708.003450768168,825.728642328667,620.924366347324,340.1717257538905,277.41942656975795,443.1700129229932,405.9033788686876,537.0453699710473,178.83491871461857,322.55795970881456,267.1131692116429,545.1982047225765,157.1594419923941,209.81487272817523,252.04552597527373,314.4849741172554,132.63443519466418,203.322440741794,288.5924403807146,367.53700049225904,892.2256955080009,610.0237594812804,552.8582159392115,634.5420025174582,436.0489756928017,237.38431807589214,572.5231337980626,889.7035442972268,478.1743998092078,615.3768797322983,905.04904596081},
                {3154.876135106994,2170.4361967020036,6404.346745071244,4707.509719441287,5764.37882385239,4921.176234560454,3113.3093811822605,520.0964975209806,896.2410867352797,1332.2594482796521,2047.8142140865846,1139.473571608747,740.4282160734017,821.2237964807173,1186.7500251611943,2028.5591312421213,521.391821483499,580.6745827035262,992.2945990744025,1538.5816610734653,220.74832416190233,764.3386079934977,532.861643579607,321.1017035333605,217.2206316666227,483.33788440440713,386.0828308744428,428.29283360770546,263.5282839433092,95.66053679647456,372.9534653459307,605.7801537677632,197.11804697229084,325.17477075177845,345.0190159018429,350.7846474228556,160.85726886236517,225.16196637949665,205.24587282280274,247.67449695194858,991.9260273714602,439.64180151821694,450.1677594115933,719.1892641664481,336.7988005646351,134.63749587230646,479.417897824755,734.7733289162828,466.0626275122039,620.6563566552328,876.7066204215866},
                {3401.7165854326495,738.8206183526622,6167.674025956539,2629.9846858977394,5066.113572235965,3833.1537421232233,3219.675989240578,885.1356521340089,1056.8680483579592,1204.457817436346,1716.3382315197662,1548.4271228628702,793.536935631278,796.3404783681422,1537.2403094690178,2017.5869374788049,449.92808978818744,533.5695097999144,575.3474321496607,1501.7876730333637,238.96294317645157,595.3297961053177,374.63170848297426,395.24932683835885,210.2535634604273,506.7728819519834,518.1769050989358,357.1380801567872,340.74448403384804,186.72583356999866,581.3005985140312,700.6947205652605,349.56546156088626,457.7891519554239,468.0140731678742,352.52530780533795,261.8775544546673,255.46552594805038,158.21174042300723,303.465950719236,1044.1948782148563,389.3705406374922,360.43061151902526,732.8827959029425,312.34097103028523,67.80278646251634,378.22538490543644,551.468946349928,523.146152095763,732.2695676978735,759.7620092930636},
                {3292.0750689854663,1191.1728340335142,6225.717880982169,1462.3558229413095,4289.591709019714,2334.3658110517763,3358.7566341888414,1773.3807255508887,1529.4618327211317,1053.4837667240356,1418.890469258986,1840.6658187780959,802.9079022707668,765.6959164564239,1806.4768774866193,1946.9876591742868,557.9772454952708,528.1633810405142,381.7176383350742,1415.7620389583126,628.9041086503574,592.794422726502,354.0168941755063,385.20822773793327,179.77455149666633,536.7273922693862,734.161090939035,417.49728161652257,481.8521765784758,492.2246065080117,768.6301522888708,780.9375377290319,536.8211188870724,586.0539303478918,529.2879672678222,337.41205289097957,352.0376594994677,311.7048731109766,175.08484238671244,485.3371430224761,1115.083190011436,434.3253774738487,252.77683594321226,725.8542926265261,345.4980685647267,86.97201151781209,378.05207065667537,296.321805401823,609.3401523326834,807.3560296399307,669.1459915028394},
                {2930.5506753134155,2962.183228578784,6471.772800978502,2777.308078216393,3668.427381802655,764.9198755573985,3386.125241272937,2579.2961417841275,1979.8009502911166,1017.0147025981541,1254.3005958862195,2026.0562010638255,834.1833941882423,818.5762295678504,1880.230579301122,1832.2700154802376,712.1727455714854,555.469179573185,536.0816620244457,1350.5670677054652,914.3279276788312,619.107174029839,413.1955404215167,384.0886409606819,42.74466836800401,647.7163543210236,952.054649095134,602.9852769803616,671.6925959460216,844.7337438903367,941.4634080114657,765.0752041713833,685.1789237484271,666.6024072952849,589.0777094977136,215.99232313677635,463.90121484002526,406.4266816341143,285.82337137866926,698.6013784113927,1181.1843816941773,516.7016780448718,159.33088044770795,722.6667501341906,364.24796926987943,149.11836080100326,393.7337864349073,97.45100633304305,637.62114394511,859.067447190087,566.8293213534371},
                {2438.995118005983,4449.686241535468,6777.674569038674,4047.702970518234,3115.1069629686417,632.6870047268496,3359.46773556904,3021.1624715564344,2395.9832353648285,1016.0419287331919,1183.7480809460787,2168.7034436210724,863.5208399839308,887.5767477095835,1870.7548789654886,1594.5618486234985,791.1202118568357,542.9720713616022,802.3536320051372,1224.0032929842705,1049.9490708145283,676.4765291554576,526.2907226381103,340.29287468235844,66.27925694523087,729.8430111878747,1133.4833070085645,859.8645892903465,924.0392174787759,1158.9838605188866,1068.8146033994449,614.3816148676889,802.9267189847342,653.9443372396844,648.625283727056,30.872492642455065,535.4883348534673,492.5729291669665,474.0324905439769,862.6000386459485,1189.09348875052,631.183441180512,211.3460185395673,678.3503063904786,383.8535679383692,172.48571194342657,410.9868421867132,161.9480028024304,639.0721830745192,828.6164302342718,484.4911708484721},
                {1997.245255650622,5419.869302659363,7053.159536473445,4442.346367161996,2712.7128685871603,1496.2126146299934,3256.5205524525823,3132.586861477369,2515.8276072741105,1172.104503313151,1379.9319007077368,1993.3222504448538,1085.6438556340747,1012.9593863530026,1676.5277823613055,1269.7037602070802,772.0904297618125,559.4146928234371,920.4740536889454,1111.7182974581885,956.9092553359898,779.2880366954683,577.790240944201,319.73563228787816,117.17625468381338,850.1234093987698,1207.11543823817,1049.5435035035941,1206.1351572519275,1343.8022897309836,1193.9510037293535,313.1179934859032,907.6462200940299,520.1005671799378,688.7543853828445,130.57936349155935,568.189131754754,589.3578343215761,550.377647950983,962.5543248420488,1137.2572077641496,695.9217849053125,344.6938089708937,670.9857328088489,323.4210061396582,191.7087012451592,406.448869292438,358.98484005714835,590.4131226310637,760.3512223835532,378.52565800982643},
                {1780.9732110573275,5923.660703593998,6890.322752196075,4235.797264360212,2168.2831900011893,2002.640404243293,2947.518999925991,2977.7190412493956,2273.6474896443133,1469.43666087784,1155.4746322779004,2050.6878854077963,924.7904414482907,1201.5572414176772,1336.5565169835288,1020.2078402241023,619.7016869305138,541.88171073796,936.8078592711548,921.5189792519127,809.3979578791574,742.8427675899924,614.9044272069439,354.12864190793596,280.4211483325273,976.7147258675907,1140.7117115020815,1144.118646780996,1325.6236505380732,1469.219139336236,1210.8718959784273,178.66167978965183,999.7954735688323,227.08564477638657,799.8107433014816,340.2413099223708,520.3523478927667,640.2486804572426,575.0578731131503,961.0732379399531,1013.2358912055174,711.765723493009,513.8489524209243,620.9031437699228,264.70474013856057,129.66741129447908,432.4050818793737,480.61291136318215,504.1137147179859,636.9407606505428,289.2466595638929},
                {1819.3405814026587,5863.088307065577,6404.68616725576,3544.8680062357894,1435.6154166331046,2144.305301283018,2550.519531871131,2413.5984761026266,1985.2351920112148,1482.7331052363018,876.9968011074889,1983.1682408815748,757.2522672377602,1260.9089453122926,1163.2684409271283,804.2710959981987,540.5150865901225,548.1401478254544,831.6792640715729,699.6242912926884,647.6880544897734,614.3055699758113,580.236320862965,476.8823206102552,445.3494261971794,1112.3591933334803,950.9642023782692,1107.5978956100773,1371.6646480371098,1359.466634361288,1243.9049760515873,515.4510896610071,1118.8570589473518,240.5756109993505,960.9803518941511,443.91998205615374,458.017210979793,624.0434395693871,513.3788636238329,878.7915611173822,834.3665213989983,646.7984728273725,642.7980193395441,552.6629751225453,192.68193302172983,45.99775867605989,425.57544052768856,575.8063680721673,336.16890862896906,548.58471333938,172.4956344499512},
                {1568.2417734906871,5627.427796609242,5572.0245392497545,2681.5593814296053,683.5294471225393,1923.232196616826,2074.4167942041345,1801.4453047267311,1322.9448599518944,1550.7316202415534,404.1700461318832,1898.6096452696197,712.8894251504131,1334.6883654691749,1156.8536099862326,784.6143339991276,634.2214930122193,554.3019553291139,638.7988677382281,537.5210641078186,473.9568720580721,473.93299467324823,533.5912009365906,575.3435358873643,647.0241293971844,1263.8414638620643,618.1279758281096,955.9660370943786,1320.2152161643533,1042.8060512685456,1267.5361324095275,800.1140309015311,1270.3438277985072,736.8597099906377,1114.0294665403162,509.71039104738554,381.18139252987055,588.7264362395663,384.3449787259626,729.9266203200428,614.4342801172317,547.9172119748798,663.5053161881574,517.1911994243555,144.42935393780158,64.09437101366173,394.76629915098874,628.8792330279822,160.37276467045817,446.6918762908014,134.03344874511095},
                {1080.2767473134,5281.95177382304,4950.626620582797,2096.1100512066077,590.1649484733452,1457.4126889539011,1545.3159932359156,1289.0183710149004,588.5131857673074,1488.9781662725825,374.3765412272261,1804.2045726180047,993.6615692893232,1438.498418733235,1227.4466151870618,1014.6158853385645,772.8199820466166,440.9867392090354,406.26756735553147,556.0053371486889,533.6648913059846,295.4554304989501,429.74792665273105,640.4744950999319,902.2925928232554,1335.7671937406167,304.75610057817505,707.0204524709833,1177.6964643220597,693.4740736572616,1167.9254551589402,982.7659020545126,1356.667090172044,1181.096625608004,1302.3634422607447,536.1094990409927,403.3703611922251,502.41033021787484,219.35297038167604,606.1390686502724,367.73944267606123,340.4514791912103,681.6873675527344,434.79892909864054,54.69688976446764,128.34396007663034,350.8594974683319,636.2290519588543,109.35669813843313,394.11245347692756,199.86680553507603},
                {736.43048772352,5239.404494466389,4852.827482887929,2032.3152414096908,1373.8947670154816,958.307908015344,1259.7115786879745,961.1397696026899,299.1237252101091,1466.5412690964004,695.1075353574515,1718.5627706756864,1368.7940175390825,1470.711513120007,1282.1154234336639,1260.7605698589969,882.3531763990643,359.9970373012316,161.9112099712234,608.4182094213579,624.841557743911,337.8520193478363,422.88361578970057,546.2183462400843,1161.9409876986476,1337.8795078003948,158.71197341078104,496.1245797039377,1003.6770945995814,287.2961136769214,1101.7602222226885,886.6053227102628,1451.0223543840327,1483.0029819254285,1425.5186404738397,616.3512909108966,473.46780724786237,431.5292242922777,101.3956436149511,493.62898160743714,176.88381898822175,156.94453364646603,577.5222271897892,451.0102598782114,115.91870532109773,140.18169415410318,313.4393520028337,611.6102606721495,257.03371898049664,356.84508814141327,269.64492008244144},
                {1413.0486152930907,5522.843769627263,5255.994922632989,2782.358239160508,2114.762218327591,246.1857331172854,1163.7118632880006,1023.2109314555681,1065.2045888217465,1557.8003728999606,803.5103913593831,1594.2167168636984,1644.6615011532015,1516.3110660158684,1134.473410604493,1479.4976259730638,926.6106194390627,161.23246356526354,241.54273204815436,783.7659415364122,737.6137646599417,397.7123606165665,444.29747943238976,409.9728894296465,1277.1197264532047,1404.3319068155256,250.96861880919153,360.88525315931975,837.3655377266995,372.0836748916001,1016.3035166755998,653.9430755715997,1426.6297393330865,1639.6510322948895,1425.8535167217008,724.7699071449482,580.3585526230505,390.33188217180503,71.70978356708211,402.9891903481842,57.91977852975923,58.0114916292902,503.7460434078068,407.66514044564735,212.17440265412304,166.6955133123506,296.3896332441231,574.6778173227226,353.0830876307762,305.57946301876353,317.10994237313287},
                {2374.8688860666393,5921.923907082665,5629.049040611796,3760.2330930722524,2676.367368874354,346.77181347429786,1123.011475225806,1470.9426370614117,1810.2749028970857,1530.7876600933346,787.368744038843,1312.3600398452243,1820.0770988500205,1399.0120667011543,893.5771746807004,1543.4665948552183,1014.7404022557953,140.64642385797967,466.28902309495044,914.2906239266323,854.300934545648,343.0110009431163,475.1526932869686,141.16453471112342,1455.889347624713,1320.9803863000905,446.675940747781,332.65629293669025,722.5541023168626,617.8646312752568,1016.4259229958845,440.3447600255432,1281.4673816197958,1592.8525585317896,1353.166128218527,725.4009050814411,662.9386333793922,427.0333108173557,69.68629234844705,365.3999625815343,105.09619658901823,128.4769554959008,416.5718443002779,361.3198984447849,294.3577552031987,201.04758466918116,310.7813618791781,524.1549788568769,385.58417943750015,236.86135385624434,329.58403810687435},
                {3656.1333064594855,6170.08431866353,5463.537949217067,4678.223079582542,3080.744514692027,786.0776397360122,1067.4996531710701,1957.714818109862,2307.500307639543,1611.269692249729,435.8512395488708,1093.1471983861572,1728.8219428192783,1362.069010216311,686.791617931143,1470.914032952858,1048.6246418303324,358.6304561529506,695.2280353660093,1125.5484695575276,918.0962213548474,203.62147874624878,536.7789586087123,348.99264574976587,1512.4645459639144,1231.4493419599291,553.3906933094161,470.480105099842,597.1575205880299,797.7831443300687,1047.2410110788433,494.47408952892977,1078.5950773050977,1354.28516938933,1210.274726875947,646.2380364616624,667.8575960515806,509.68250021677903,177.79977929305267,370.84052650662187,156.25409000575328,196.86219541177095,338.83074170791025,247.97968605453605,392.4444930767941,297.4893503588633,286.2598591365529,505.6288831448793,315.8399625659644,173.20120815270008,325.41402332306944},
                {5026.250114754332,5915.517884764837,4740.335643383183,5083.150876700095,3688.9687728256645,976.9076570710843,1057.5879784696392,2302.4393140314173,2591.130120174288,1733.186846405739,355.9565503210072,724.7464762398445,1572.6235918649377,1362.4400009681015,686.2587457430661,1290.0019768268055,1018.0928918092997,623.4883245926823,849.3459007284921,1277.0853142332842,967.6655392692833,16.91652653594818,572.3061669775761,678.1786656532465,1535.3047802558622,1037.12740604063,633.1914372869705,684.7364900022579,473.1754762821489,901.2243300502267,1148.9014636022987,712.0785137623449,775.4200773730288,1093.5468839754358,980.7432785236836,469.92442753980504,627.4140613773926,570.8921062167153,311.2058316067873,376.15087104975976,193.5423194557229,187.69370166636725,321.7481102471945,73.69164033469426,462.56936195245333,375.9474890452803,276.8464690695973,413.3322640645539,284.7380607755944,104.6024678842007,273.3808211813426},
                {6135.7568276525335,5304.432699717564,3444.369265617596,5059.162178587369,4087.068930345507,1362.3665841311747,1143.890231799314,2613.1525741206938,2503.640027275803,1922.482144350062,700.7151796705847,403.1803719363274,1383.4344399496226,1450.3547160666333,875.7251651521234,1003.8151528495134,918.1387040827731,774.4147317504329,940.5784413142449,1328.2648833801736,1004.1866869359695,171.3243037963933,548.5042854472173,971.2620220445074,1506.3077757119981,794.5029397776224,657.4875950523327,930.6092200109433,466.084533347915,967.3788477564244,1320.710689865816,805.2326375798284,512.8767025453305,803.5798152719204,785.7978829567164,285.1388953418928,484.3444019523533,634.9858728551005,445.76632296947315,353.1374795191516,209.48368439068352,165.17976089227255,279.3884841231572,63.02812096132058,504.0698679205089,482.4623705273276,210.96076723321735,354.0785631831769,248.40781928776985,53.37952249467617,252.35992621564787},
                {6734.7340003597965,4465.834925657012,2055.4337770760576,4364.554134541243,4420.648999482229,1964.064075812053,1665.7412831876088,2578.2585846220486,2186.8491845922154,2042.6991229195737,1092.1941312439355,424.472151139931,1217.8020424854544,1538.8438136563427,1016.856797090871,713.3587278178514,688.1644155027806,885.4723772576893,857.2696998029596,1321.5635719747488,981.6129979947141,317.74168362729966,481.7690475094249,1175.8199977553122,1395.0348571666618,619.7069235634772,650.6528385904725,1124.9213069950881,661.1974547519018,1117.7991574095515,1428.8663042421338,734.5447407009307,343.015430009012,619.5383075197759,663.125843614683,189.32968636754276,299.02198848432465,661.8319708298301,558.1379467660001,281.97863958376155,209.5672086130417,139.3188109587527,211.81499431171915,214.73963382815953,533.6159603077865,533.27525805095,198.60225290178633,352.8890583506415,231.75192808073265,34.5761790992945,239.74371637998317}
                };

    this->setData(fftdata);
    
    std::vector<std::string> arguments_initialize = {"pyfuncs","initialize"};
    std::vector<PyObject*> pyArgs = {this->getModel(),
                this->getData()};
    this->callPythonFunction(arguments_initialize, pyArgs);

    std::vector<std::string> arguments_predict = {"pyfuncs","predict"};
    pyArgs = {this->getModel(), 
                                    this->getFeats(), 
                                    this->getScaler(),
                                    this->getData()};
    this->callPythonFunction(arguments_predict, pyArgs);

    

}

void lfpInferenceEngine::printInPython(void) {
    PyRun_SimpleString("print('Python session still on')");
    return;
}


int lfpInferenceEngine::callPythonFunction(vector<string> args, vector<PyObject*> pyArgs = {}) {

    int length = args.size();
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    if (length < 2) {
        fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
        return 1;
    }

	pName = PyUnicode_DecodeFSDefault(args[0].c_str());
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, args[1].c_str());
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            if (pyArgs.empty()) {
                pArgs = PyTuple_New(length - 2);
                for (i = 0; i < length - 2; ++i) {
                    //pValue = PyLong_FromLong(atoi(args[i + 2].c_str()));
                    pValue = PyUnicode_FromString(args[i + 2].c_str());
                    if (!pValue) {
                        Py_DECREF(pArgs);
                        Py_DECREF(pModule);
                        fprintf(stderr, "Cannot convert argument\n");
                        return 1;
                    }
                    /* pValue reference stolen here: */
                    PyTuple_SetItem(pArgs, i, pValue);
                }
            }
            else {
                pArgs = PyTuple_New(pyArgs.size());
                for (i = 0; i < pyArgs.size(); i++) {
                    PyTuple_SetItem(pArgs, i, pyArgs[i]);
                }
            }
            
            pValue = PyObject_CallObject(pFunc, pArgs);
            //Py_DECREF(pArgs);
            if (pValue != NULL) {

                this->pResult = Py_NewRef(pValue);

                Py_DECREF(pValue);
                //Py_DECREF(pResult);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", args[1].c_str());
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", args[0].c_str());
        return 1;
    }
    //return pValue;
    return 0;

}

// Add FFT vector to back of data vector and erase first FFT
void lfpInferenceEngine::pushFFTSample(std::vector<double> fft) {
    //std::cout << "fft size: " << fft.size() << std::endl;
    fftdata.push_back(fft);
    //if (fftdata.size() > N) {
        // cut it to size
    fftdata.erase(fftdata.begin());
    /*for (int i=0; i<fftdata.size(); i++)
    {
        std::cout << "fftdata size: " << fftdata[i].size() << std::endl;
    }
    //}*/
    this->setData(fftdata);
}

void lfpInferenceEngine::setModel(PyObject *newModel) {
    pModel = newModel;
}

void lfpInferenceEngine::setFeats(PyObject *newFeats) {
    pFeats = newFeats;
}

void lfpInferenceEngine::setScaler(PyObject *newScaler) {
    pScaler = newScaler;
}

void lfpInferenceEngine::setData(std::vector<std::vector<double>> newData) {
    PyObject* outer_list = PyList_New(newData.size());
    for (std::size_t i = 0; i < newData.size(); i++) {
        std::vector<double> inner_vector = newData[i];
        PyObject* inner_list = PyList_New(inner_vector.size());
        for (std::size_t j = 0; j < inner_vector.size(); j++) {
            PyList_SetItem(inner_list, j, Py_BuildValue("d", inner_vector[j]));
        }
        PyList_SetItem(outer_list, i, inner_list);
    }
    pData = outer_list;
}

void lfpInferenceEngine::load() {
    if (!PyTuple_Check(pResult)) {
        printf("pResult is not a tuple! Make sure to call callPythonFunction correctly before loading.\n");
        return;
    }

    if (!PyArg_ParseTuple(pResult,"OOO", &pModel, &pFeats, &pScaler)) {
        printf("Failed to unpack tuple.\n");
        return;
    }
}

void lfpInferenceEngine::load_data() {
    if (PyTuple_Check(pResult)) {
        printf("pResult is a tuple! Make sure to call callPythonFunction correctly before loading data.\n");
        return;
    }

    pData = pResult;
}

std::vector<int> lfpInferenceEngine::predict() {
    std::vector<std::string> arguments_predict = {"pyfuncs","predict"};
    std::vector<PyObject*> pyArgs = {pModel,
                                    pFeats,
                                    pScaler,
                                    pData};
    
    this->callPythonFunction(arguments_predict, pyArgs);

    return this->PyList_toVecInt(pResult);
}

std::vector<int> lfpInferenceEngine::PyList_toVecInt(PyObject* py_list) {
  if (PySequence_Check(py_list)) {
    PyObject* seq = PySequence_Fast(py_list, "expected a sequence");
    if (seq != NULL){
      std::vector<int> my_vector;
      my_vector.reserve(PySequence_Fast_GET_SIZE(seq));
      for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq,i);
        if(PyNumber_Check(item)){
          Py_ssize_t value = PyNumber_AsSsize_t(item, PyExc_OverflowError);
          if (value == -1 && PyErr_Occurred()) {
            //handle error
          }
          my_vector.push_back(value);
        } else {
          //handle error
        }
      }
      Py_DECREF(seq);
      return my_vector;
    } else {
      //handle error
    }
  }else{
    //handle error
  }
}

void lfpInferenceEngine::reportFFTdata() {
    for (std::size_t i = 0; i < fftdata.size(); i++) {
        std::cout << std::endl;
        std::vector<double> inner_vector = fftdata[i];
        for (std::size_t j = 0; j < inner_vector.size(); j++) {
            std::cout << inner_vector[j] << ",";
        }
    }
}