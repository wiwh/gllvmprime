* Command file to read ASCII data file into SPSS .
* Note: change location of input file to match your location .
FILE HANDLE H16D_R /name = 'c:\hrs2016\data\H16D_R.da' LRECL = 299.
DATA LIST FILE= H16D_R /
   HHID 1- 6 (A)
   PN 7- 9 (A)
   PSUBHH 10- 10 (A)
   OSUBHH 11- 11 (A)
   PPN_SP 12- 14 (A)
   PCSR 15- 15 
   PFAMR 16- 16 
   PFINR 17- 17 
   PD190 18- 18 
   PD290 19- 19 
   PD101 20- 20 
   PD102 21- 21 
   PD103 22- 22 
   PD104 23- 24 
   PD182M1 25- 26 
   PD182M2 27- 28 
   PD182M3 29- 30 
   PD182M4 31- 32 
   PD182M5 33- 34 
   PD182M6 35- 36 
   PD182M7 37- 38 
   PD182M8 39- 40 
   PD182M9 41- 42 
   PD182M10 43- 44 
   PD182M11 45- 46 
   PD182M12 47- 48 
   PD182M13 49- 50 
   PD182M14 51- 52 
   PD182M15 53- 54 
   PD182M16 55- 56 
   PD182M17 57- 58 
   PD182M18 59- 60 
   PD174 61- 62 
   PD175 63- 64 
   PD176 65- 66 
   PD177 67- 67 
   PD108M1 68- 68 
   PD108M2 69- 69 
   PD108M3 70- 70 
   PD108M4 71- 71 
   PD110 72- 72 
   PD111 73- 73 
   PD112 74- 74 
   PD113 75- 75 
   PD114 76- 76 
   PD115 77- 77 
   PD116 78- 78 
   PD117 79- 79 
   PD120 80- 80 
   PD122 81- 81 
   PD124 82- 82 
   PD125 83- 83 
   PD127 84- 84 
   PD129 85- 85 
   PD142 86- 88 
   PD143 89- 91 
   PD144 92- 94 
   PD145 95- 97 
   PD146 98- 100 
   PD189 101- 101 
   PD183M1 102- 103 
   PD183M2 104- 105 
   PD183M3 106- 107 
   PD183M4 108- 109 
   PD183M5 110- 111 
   PD183M6 112- 113 
   PD183M7 114- 115 
   PD183M8 116- 117 
   PD183M9 118- 119 
   PD183M10 120- 121 
   PD183M11 122- 123 
   PD183M12 124- 125 
   PD183M13 126- 127 
   PD183M14 128- 129 
   PD183M15 130- 131 
   PD183M16 132- 133 
   PD183M17 134- 135 
   PD183M18 136- 137 
   PD183M19 138- 139 
   PD184 140- 141 
   PD185 142- 143 
   PD186 144- 145 
   PD187 146- 146 
   PD191 147- 147 
   PD150 148- 148 
   PD151 149- 149 
   PD152 150- 150 
   PD153 151- 151 
   PD154 152- 152 
   PD155 153- 153 
   PD156 154- 154 
   PD157 155- 155 
   PD158 156- 156 
   PD159 157- 157 
   PD161 158- 158 
   PD163 159- 159 
   PD165 160- 160 
   PD167 161- 161 
   PD169 162- 162 
   PD178 163- 169 
   PD179 170- 178 
   PD180 179- 184 
   PD194 185- 185 
   PD196 186- 187 
   PD197 188- 188 
   PD198 189- 190 
   PD199 191- 191 
   PD245M1 192- 192 
   PD245M2 193- 193 
   PD245M3 194- 194 
   PD245M4 195- 195 
   PD200 196- 196 
   PD240 197- 197 
   PD241 198- 198 
   PD242 199- 199 
   PD201 200- 200 
   PD202 201- 201 
   PD203 202- 202 
   PD204 203- 203 
   PD205 204- 204 
   PD206 205- 205 
   PD207 206- 206 
   PD208 207- 207 
   PD209 208- 208 
   PD210 209- 209 
   PD211 210- 210 
   PD212 211- 211 
   PD213 212- 212 
   PD214 213- 213 
   PD215 214- 214 
   PD221 215- 215 
   PD222 216- 216 
   PD223 217- 217 
   PD224 218- 218 
   PD225 219- 219 
   PD226 220- 220 
   PD227 221- 221 
   PD228 222- 222 
   PD229 223- 223 
   PD230 224- 224 
   PD231 225- 225 
   PD232 226- 226 
   PD233 227- 227 
   PD234 228- 228 
   PD235 229- 229 
   PD216 230- 230 
   PD217 231- 231 
   PNSSCORE 232- 234 
   PNSSCORESE 235- 237 
   PD170 238- 239 
   PD172 240- 240 
   PD171 241- 241 
   PD501 242- 242 
   PD502 243- 243 
   PD505 244- 244 
   PD506 245- 245 
   PD507 246- 246 
   PD508 247- 247 
   PD509 248- 248 
   PD510 249- 249 
   PD511 250- 250 
   PD512 251- 251 
   PD513 252- 252 
   PD514 253- 253 
   PD515 254- 254 
   PD516 255- 255 
   PD517 256- 256 
   PD518 257- 257 
   PD519 258- 258 
   PD520 259- 259 
   PD521 260- 260 
   PD522 261- 261 
   PD523 262- 262 
   PD524 263- 263 
   PD525 264- 264 
   PD526 265- 265 
   PD527 266- 266 
   PD528 267- 267 
   PD529 268- 268 
   PD530 269- 269 
   PD531 270- 270 
   PD532 271- 271 
   PD533 272- 272 
   PD534 273- 273 
   PD535 274- 274 
   PD536 275- 275 
   PD537 276- 276 
   PD538 277- 277 
   PD539 278- 278 
   PD540 279- 279 
   PD541 280- 280 
   PD542 281- 281 
   PD543 282- 282 
   PD544 283- 283 
   PD545 284- 284 
   PD546 285- 285 
   PD547 286- 286 
   PD548 287- 287 
   PD549 288- 288 
   PD550 289- 289 
   PD551 290- 290 
   PD552 291- 291 
   PD553 292- 292 
   PD554 293- 293 
   PD555 294- 294 
   PD556 295- 295 
   PD557 296- 296 
   PVDATE 297- 298 
   PVERSION 299- 299 
.

VARIABLE LABELS
   HHID"HOUSEHOLD IDENTIFICATION NUMBER"
   PN"RESPONDENT PERSON IDENTIFICATION NUMBER"
   PSUBHH"2016 SUB HOUSEHOLD IDENTIFICATION NUMBER"
   OSUBHH"2014 SUB HOUSEHOLD IDENTIFICATION NUMBER"
   PPN_SP"2016 SPOUSE/PARTNER PERSON NUMBER"
   PCSR"2016 WHETHER COVERSHEET RESPONDENT"
   PFAMR"2016 WHETHER FAMILY RESPONDENT"
   PFINR"2016 WHETHER FINANCIAL RESPONDENT"
   PD190"ALTWAVE FLAG FOR D159 AND D178 SEQUENCES"
   PD290"ALTWAVE FLAG FOR NUMBER SERIES"
   PD101"RATE MEMORY"
   PD102"RATE MEMORY PAST"
   PD103"WORDS PREAMBLE"
   PD104"D104 WORD LIST ASSIGNMENT"
   PD182M1"WORD RECALL IMMEDIATE - 1"
   PD182M2"WORD RECALL IMMEDIATE - 2"
   PD182M3"WORD RECALL IMMEDIATE - 3"
   PD182M4"WORD RECALL IMMEDIATE - 4"
   PD182M5"WORD RECALL IMMEDIATE - 5"
   PD182M6"WORD RECALL IMMEDIATE - 6"
   PD182M7"WORD RECALL IMMEDIATE - 7"
   PD182M8"WORD RECALL IMMEDIATE - 8"
   PD182M9"WORD RECALL IMMEDIATE - 9"
   PD182M10"WORD RECALL IMMEDIATE - 10"
   PD182M11"WORD RECALL IMMEDIATE - 11"
   PD182M12"WORD RECALL IMMEDIATE - 12"
   PD182M13"WORD RECALL IMMEDIATE - 13"
   PD182M14"WORD RECALL IMMEDIATE - 14"
   PD182M15"WORD RECALL IMMEDIATE - 15"
   PD182M16"WORD RECALL IMMEDIATE - 16"
   PD182M17"WORD RECALL IMMEDIATE - 17"
   PD182M18"WORD RECALL IMMEDIATE - 18"
   PD174"NUMBER GOOD - IMMEDIATE"
   PD175"NUMBER WRONG - IMMEDIATE"
   PD176"NUMBER FORGOTTEN - IMMEDIATE"
   PD177"NONE REMEMBERED - IMMEDIATE - FLAG"
   PD108M1"D108M IWER CHECKPOINT -1"
   PD108M2"D108M IWER CHECKPOINT -2"
   PD108M3"D108M IWER CHECKPOINT -3"
   PD108M4"D108M IWER CHECKPOINT -4"
   PD110"FEELING DEPRESSED W/IN PREV WK"
   PD111"FELT ACTIVITIES WERE EFFORTS"
   PD112"WAS SLEEP RESTLESS W/IN PREV WK"
   PD113"WAS R HAPPY W/IN PREV WK"
   PD114"LONELINESS FELT W/IN PREV WK"
   PD115"ENJOYED LIFE W/IN PREV WK"
   PD116"FELT SAD W/IN PREV WK"
   PD117"FELT UNMOTIVATED W/IN PREV WK"
   PD120"COUNT 20 - FIRST TRY"
   PD122"INTRO-END 1ST TRY CNT BACKWARDS"
   PD124"IWER CHECK 20-1ST TRY"
   PD125"INTRO COUNT BACKWARDS 2ND TRY"
   PD127"INTRO END CNT BACKWARDS 2ND TRY"
   PD129"IWER CHECK 20- SECOND TRY"
   PD142"SERIES MINUS 7- 1"
   PD143"SERIES MINUS 7- 2"
   PD144"SERIES MINUS 7- 3"
   PD145"SERIES MINUS 7- 4"
   PD146"SERIES MINUS 7- 5"
   PD189"D189 IWER CHECKPOINT"
   PD183M1"WORD RECALL DELAYED - 1"
   PD183M2"WORD RECALL DELAYED - 2"
   PD183M3"WORD RECALL DELAYED - 3"
   PD183M4"WORD RECALL DELAYED - 4"
   PD183M5"WORD RECALL DELAYED - 5"
   PD183M6"WORD RECALL DELAYED - 6"
   PD183M7"WORD RECALL DELAYED - 7"
   PD183M8"WORD RECALL DELAYED - 8"
   PD183M9"WORD RECALL DELAYED - 9"
   PD183M10"WORD RECALL DELAYED - 10"
   PD183M11"WORD RECALL DELAYED - 11"
   PD183M12"WORD RECALL DELAYED - 12"
   PD183M13"WORD RECALL DELAYED - 13"
   PD183M14"WORD RECALL DELAYED - 14"
   PD183M15"WORD RECALL DELAYED - 15"
   PD183M16"WORD RECALL DELAYED - 16"
   PD183M17"WORD RECALL DELAYED - 17"
   PD183M18"WORD RECALL DELAYED - 18"
   PD183M19"WORD RECALL DELAYED - 19"
   PD184"NUMBER GOOD - DELAYED"
   PD185"NUMBER WRONG - DELAYED"
   PD186"NUMBER FORGOTTEN - DELAYED"
   PD187"NONE REMEMBERED - DELAYED"
   PD191"WORDLIST CHECK DID R USE AID"
   PD150"COGNITION INTRO"
   PD151"TODAYS DATE- MONTH"
   PD152"TODAYS DATE- DAY"
   PD153"TODAYS DATE- YEAR"
   PD154"TODAYS DATE- DAY OF WEEK"
   PD155"TOOL USED TO CUT PAPER"
   PD156"NAME OF PRICKLY DESERT PLANT"
   PD157"WHO IS THE PRESIDENT OF US"
   PD158"WHO IS THE VICE-PRESIDENT OF US"
   PD159"D159 CONTINUE IW- VOCAB WORDS"
   PD161"MEANING OF REPAIR/CONCEAL"
   PD163"MEANING OF FABRIC/ENORMOUS"
   PD165"MEANING OF DOMESTIC/PERIMETER"
   PD167"MEANING OF REMORSE/COMPASSION"
   PD169"MEANING OF PLAGIARIZE/AUDACIOUS"
   PD178"CHANCE GET DISEASE"
   PD179"LOTTERY SPLIT 5 WAYS"
   PD180"INTEREST ON SAVINGS"
   PD194"INTRO TO ANIMALS"
   PD196"TOTAL ANIMALS ANSWERS"
   PD197"ANIMAL MISTAKES"
   PD198"ANIMAL MISTAKES NUMBER"
   PD199"TIMING TOOL USED"
   PD245M1"ANIMAL NAME PROBLEMS -1"
   PD245M2"ANIMAL NAME PROBLEMS -2"
   PD245M3"ANIMAL NAME PROBLEMS -3"
   PD245M4"ANIMAL NAME PROBLEMS -4"
   PD200"INTRO-QUANTITATIVE NUMBER SERIES"
   PD240"INTRO-QUANTITATIVE NUMBER SERIES"
   PD241"INTRO-QUANTITATIVE NUMBER SERIES"
   PD242"INTRO-QUANTITATIVE NUMBER SERIES"
   PD201"NUMBER SERIES-G1"
   PD202"NUMBER SERIES-H1"
   PD203"NUMBER SERIES-I1"
   PD204"NUMBER SERIES-A1"
   PD205"NUMBER SERIES-B1"
   PD206"NUMBER SERIES-C1"
   PD207"NUMBER SERIES-D1"
   PD208"NUMBER SERIES-E1"
   PD209"NUMBER SERIES-F1"
   PD210"NUMBER SERIES-J1"
   PD211"NUMBER SERIES-K1"
   PD212"NUMBER SERIES-L1"
   PD213"NUMBER SERIES-M1"
   PD214"NUMBER SERIES-N1"
   PD215"NUMBER SERIES-O1"
   PD221"NUMBER SERIES-G2"
   PD222"NUMBER SERIES-H2"
   PD223"NUMBER SERIES-I2"
   PD224"NUMBER SERIES-A2"
   PD225"NUMBER SERIES-B2"
   PD226"NUMBER SERIES-C2"
   PD227"NUMBER SERIES-D2"
   PD228"NUMBER SERIES-E2"
   PD229"NUMBER SERIES-F2"
   PD230"NUMBER SERIES-J2"
   PD231"NUMBER SERIES-K2"
   PD232"NUMBER SERIES-L2"
   PD233"NUMBER SERIES-M2"
   PD234"NUMBER SERIES-N2"
   PD235"NUMBER SERIES-O2"
   PD216"NUMBER OF ITEMS CORRECT"
   PD217"NUMBER OF ITEMS REFUSED"
   PNSSCORE"CALCULATED NUMBER SERIES SCORE"
   PNSSCORESE"STANDARD ERROR OF NUMBER SERIES SCORE"
   PD170"TICS SCORE COUNT"
   PD172"D172 FLAG ASSIST - D"
   PD171"ASSIST SECTION D - COGNITIVE"
   PD501"RATE MEMORY- PC"
   PD502"COMPARE MEM TO PREV WAVE- PC"
   PD505"MEM/INTELLIGENCE INTRO- P C"
   PD506"RATE R AT REMEMBERING THINGS- PC"
   PD507"ORGANIZATION IMPROVED- PC"
   PD508"ORGANIZATION WORSE- PC"
   PD509"RATE R AT REMEMBERING RECENT EVENTS- PC"
   PD510"REMEMBERING RECENT EVENTS IMPROVED- PC"
   PD511"REMEMBERING RECENT EVENTS WORSE- PC"
   PD512"RATE R AT CONVERSATION RECALL- PC"
   PD513"CONVERSATION RECALL IMPROVED- PC"
   PD514"CONVERSATION RECALL WORSE- PC"
   PD515"RATE REMEMBERING OWN PHONE NUM- PC"
   PD516"REMEMBERING OWN PHONE NUM IMPROVE- PC"
   PD517"REMEMBERING OWN PHONE NUM WORSE- PC"
   PD518"RATE REMEMBERING CURRENT DY/MO- PC"
   PD519"REMEMBERING CURRENT DY/MO IMPROVE- PC"
   PD520"REMEMBERING CURRENT DY/MO WORSE- PC"
   PD521"RATE REMEMBERING WHERE THINGS KEPT- PC"
   PD522"WHERE THINGS ARE KEPT IMPROVED- PC"
   PD523"WHERE THINGS ARE KEPT WORSE- PC"
   PD524"RATE FINDING THINGS IN DIFF PLACES- PC"
   PD525"FINDING THINGS IMPROVED- PC"
   PD526"FINDING THINGS WORSE- PC"
   PD527"RATE WORKING WITH FAMILIAR MACHINES- PC"
   PD528"WORKING WITH FAMILIAR MACHINES IMPR- PC"
   PD529"WORKING WITH FAMILIAR MACHINES WORSE- PC"
   PD530"RATE LEARNING NEW MACHINES- PC"
   PD531"LEARNING NEW MACHINES IMPROVED- PC"
   PD532"LEARNING NEW MACHINES WORSE- PC"
   PD533"RATE LEARNING NEW THINGS IN GENERAL- PC"
   PD534"LEARNING ABILITY IMPROVE- PC"
   PD535"LEARNING ABILITY WORSE- PC"
   PD536"RATE ABILITY TO FOLLOW STORY- PC"
   PD537"ABILITY TO FOLLOW STORY IMPROVE- PC"
   PD538"ABILITY TO FOLLOW STORY WORSE- PC"
   PD539"RATE MAKING DECISIONS- PC"
   PD540"MAKE DECISIONS IMPROVE- PC"
   PD541"MAKE DECISIONS WORSE- PC"
   PD542"RATE HANDLING SHOPPING MONEY- PC"
   PD543"HANDLING SHOPPING MONEY IMPROVE- PC"
   PD544"HANDLING SHOPPING MONEY WORSE- PC"
   PD545"RATE HANDLING FINANCES- PC"
   PD546"HANDLING FINANCES IMPROVE- PC"
   PD547"HANDLING FINANCES WORSE- PC"
   PD548"RATE HANDLING DAILY ARITHMETIC PROBS- PC"
   PD549"HANDLING ARITHMETIC PROBLEMS IMPROVE- PC"
   PD550"HANDLING  ARITHMETIC PROBLEMS WORSE- PC"
   PD551"RATE REASONING- PC"
   PD552"REASONING IMPROVE- PC"
   PD553"REASONING WORSE- PC"
   PD554"GET LOST IN FAMILIAR PLACES- PC"
   PD555"WANDER OFF- PC"
   PD556"CAN R BE LEFT ALONE- PC"
   PD557"DOES R HALLUCINATE- PC"
   PVDATE"2016 DATA MODEL VERSION"
   PVERSION"2016 DATA RELEASE VERSION"
.
execute.
save  /outfile 'c:\hrs2016\spss\H16D_R.sav'.
execute.
