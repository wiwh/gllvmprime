* Command file to read ASCII data file into SPSS .
* Note: change location of input file to match your location .
FILE HANDLE H10D_R /name = 'c:\hrs2010\data\H10D_R.da' LRECL = 302.
DATA LIST FILE= H10D_R /
   HHID 1- 6 (A)
   PN 7- 9 (A)
   MSUBHH 10- 10 (A)
   LSUBHH 11- 11 (A)
   MPN_SP 12- 14 (A)
   MCSR 15- 15 
   MFAMR 16- 16 
   MFINR 17- 17 
   MD190 18- 18 
   MD101 19- 19 
   MD102 20- 20 
   MD103 21- 21 
   MD104 22- 23 
   MD182M1 24- 25 
   MD182M2 26- 27 
   MD182M3 28- 29 
   MD182M4 30- 31 
   MD182M5 32- 33 
   MD182M6 34- 35 
   MD182M7 36- 37 
   MD182M8 38- 39 
   MD182M9 40- 41 
   MD182M10 42- 43 
   MD182M11 44- 45 
   MD182M12 46- 47 
   MD182M13 48- 49 
   MD182M14 50- 51 
   MD182M15 52- 53 
   MD174 54- 55 
   MD175 56- 57 
   MD176 58- 59 
   MD177 60- 60 
   MD108M1 61- 61 
   MD108M2 62- 62 
   MD108M3 63- 63 
   MD108M4 64- 64 
   MD188 65- 65 
   MD110 66- 66 
   MD111 67- 67 
   MD112 68- 68 
   MD113 69- 69 
   MD114 70- 70 
   MD115 71- 71 
   MD116 72- 72 
   MD117 73- 73 
   MD118 74- 74 
   MD120 75- 75 
   MD122 76- 76 
   MD124 77- 77 
   MD125 78- 78 
   MD127 79- 79 
   MD129 80- 80 
   MD130 81- 81 
   MD132 82- 82 
   MD134 83- 83 
   MD135 84- 84 
   MD137 85- 85 
   MD139 86- 86 
   MD142 87- 89 
   MD143 90- 92 
   MD144 93- 95 
   MD145 96- 98 
   MD146 99- 101 
   MD189 102- 102 
   MD183M1 103- 104 
   MD183M2 105- 106 
   MD183M3 107- 108 
   MD183M4 109- 110 
   MD183M5 111- 112 
   MD183M6 113- 114 
   MD183M7 115- 116 
   MD183M8 117- 118 
   MD183M9 119- 120 
   MD183M10 121- 122 
   MD183M11 123- 124 
   MD183M12 125- 126 
   MD183M13 127- 128 
   MD183M14 129- 130 
   MD184 131- 132 
   MD185 133- 134 
   MD186 135- 136 
   MD187 137- 137 
   MD150 138- 138 
   MD151 139- 139 
   MD152 140- 140 
   MD153 141- 141 
   MD154 142- 142 
   MD155 143- 143 
   MD156 144- 144 
   MD157 145- 145 
   MD158 146- 146 
   MD159 147- 147 
   MD161 148- 148 
   MD163 149- 149 
   MD165 150- 150 
   MD167 151- 151 
   MD169 152- 152 
   MD178 153- 162 
   MD179 163- 172 
   MD180 173- 181 
   MD194 182- 182 
   MD196 183- 184 
   MD197 185- 185 
   MD198 186- 194 
   MD199 195- 195 
   MD200 196- 196 
   MD240 197- 197 
   MD241 198- 198 
   MD242 199- 199 
   MD201 200- 200 
   MD202 201- 201 
   MD203 202- 202 
   MD204 203- 203 
   MD205 204- 204 
   MD206 205- 205 
   MD207 206- 206 
   MD208 207- 207 
   MD209 208- 208 
   MD210 209- 209 
   MD211 210- 210 
   MD212 211- 211 
   MD213 212- 212 
   MD214 213- 213 
   MD215 214- 214 
   MD221 215- 215 
   MD222 216- 216 
   MD223 217- 217 
   MD224 218- 218 
   MD225 219- 219 
   MD226 220- 220 
   MD227 221- 221 
   MD228 222- 222 
   MD229 223- 223 
   MD230 224- 224 
   MD231 225- 225 
   MD232 226- 226 
   MD233 227- 227 
   MD234 228- 228 
   MD235 229- 229 
   MNSSCORE 230- 234 (1)
   MNSSCORESE 235- 239 (1)
   MD216 240- 240 
   MD217 241- 241 
   MD170 242- 243 
   MD172 244- 244 
   MD171 245- 245 
   MD501 246- 246 
   MD502 247- 247 
   MD505 248- 248 
   MD506 249- 249 
   MD507 250- 250 
   MD508 251- 251 
   MD509 252- 252 
   MD510 253- 253 
   MD511 254- 254 
   MD512 255- 255 
   MD513 256- 256 
   MD514 257- 257 
   MD515 258- 258 
   MD516 259- 259 
   MD517 260- 260 
   MD518 261- 261 
   MD519 262- 262 
   MD520 263- 263 
   MD521 264- 264 
   MD522 265- 265 
   MD523 266- 266 
   MD524 267- 267 
   MD525 268- 268 
   MD526 269- 269 
   MD527 270- 270 
   MD528 271- 271 
   MD529 272- 272 
   MD530 273- 273 
   MD531 274- 274 
   MD532 275- 275 
   MD533 276- 276 
   MD534 277- 277 
   MD535 278- 278 
   MD536 279- 279 
   MD537 280- 280 
   MD538 281- 281 
   MD539 282- 282 
   MD540 283- 283 
   MD541 284- 284 
   MD542 285- 285 
   MD543 286- 286 
   MD544 287- 287 
   MD545 288- 288 
   MD546 289- 289 
   MD547 290- 290 
   MD548 291- 291 
   MD549 292- 292 
   MD550 293- 293 
   MD551 294- 294 
   MD552 295- 295 
   MD553 296- 296 
   MD554 297- 297 
   MD555 298- 298 
   MD556 299- 299 
   MD557 300- 300 
   MVDATE 301- 301 
   MVERSION 302- 302 
.

VARIABLE LABELS
   HHID"HOUSEHOLD IDENTIFICATION NUMBER"
   PN"RESPONDENT PERSON IDENTIFICATION NUMBER"
   MSUBHH"2010 SUB HOUSEHOLD IDENTIFICATION NUMBER"
   LSUBHH"2008 SUB HOUSEHOLD IDENTIFICATION NUMBER"
   MPN_SP"2010 SPOUSE/PARTNER PERSON NUMBER"
   MCSR"2010 WHETHER COVERSHEET RESPONDENT"
   MFAMR"2010 WHETHER FAMILY RESPONDENT"
   MFINR"2010 WHETHER FINANCIAL RESPONDENT"
   MD190"ALTWAVE FLAG FOR D159 AND D178 SEQUENCES"
   MD101"RATE MEMORY"
   MD102"RATE MEMORY PAST"
   MD103"WORDS PREAMBLE"
   MD104"D104 WORD LIST ASSIGNMENT"
   MD182M1"WORD RECALL IMMEDIATE - 1"
   MD182M2"WORD RECALL IMMEDIATE - 2"
   MD182M3"WORD RECALL IMMEDIATE - 3"
   MD182M4"WORD RECALL IMMEDIATE - 4"
   MD182M5"WORD RECALL IMMEDIATE - 5"
   MD182M6"WORD RECALL IMMEDIATE - 6"
   MD182M7"WORD RECALL IMMEDIATE - 7"
   MD182M8"WORD RECALL IMMEDIATE - 8"
   MD182M9"WORD RECALL IMMEDIATE - 9"
   MD182M10"WORD RECALL IMMEDIATE - 10"
   MD182M11"WORD RECALL IMMEDIATE - 11"
   MD182M12"WORD RECALL IMMEDIATE - 12"
   MD182M13"WORD RECALL IMMEDIATE - 13"
   MD182M14"WORD RECALL IMMEDIATE - 14"
   MD182M15"WORD RECALL IMMEDIATE - 15"
   MD174"NUMBER GOOD - IMMEDIATE"
   MD175"NUMBER WRONG - IMMEDIATE"
   MD176"NUMBER FORGOTTEN - IMMEDIATE"
   MD177"NONE REMEMBERED - IMMEDIATE - FLAG"
   MD108M1"D108M IWER CHECKPOINT -1"
   MD108M2"D108M IWER CHECKPOINT -2"
   MD108M3"D108M IWER CHECKPOINT -3"
   MD108M4"D108M IWER CHECKPOINT"
   MD188"D188 IWER CHECKPOINT"
   MD110"FEELING DEPRESSED W/IN PREV WK"
   MD111"FELT ACTIVITIES WERE EFFORTS"
   MD112"WAS SLEEP RESTLESS W/IN PREV WK"
   MD113"WAS R HAPPY W/IN PREV WK"
   MD114"LONELINESS FELT W/IN PREV WK"
   MD115"ENJOYED LIFE W/IN PREV WK"
   MD116"FELT SAD W/IN PREV WK"
   MD117"FELT UNMOTIVATED W/IN PREV WK"
   MD118"FELT FULL OF ENERGY W/IN PREV WK"
   MD120"COUNT 20 - FIRST TRY"
   MD122"INTRO-END 1ST TRY CNT BACKWARDS"
   MD124"IWER CHECK 20-1ST TRY"
   MD125"INTRO COUNT BACKWARDS 2ND TRY"
   MD127"INTRO END CNT BACKWARDS 2ND TRY"
   MD129"IWER CHECK 20- SECOND TRY"
   MD130"INTRO COUNT BACKWARDS 86-1ST TRY"
   MD132"END COUNT BACKWARDS 86-1ST TRY"
   MD134"IWER CHECK 86-1ST TRY"
   MD135"COUNT 86/ START OVER/ SECOND TRY"
   MD137"CONTINUE/START OVER/ SECOND TRY"
   MD139"CORRECT COUNT 86/ START OVER/ SECOND TRY"
   MD142"SERIES MINUS 7- 1"
   MD143"SERIES MINUS 7- 2"
   MD144"SERIES MINUS 7- 3"
   MD145"SERIES MINUS 7- 4"
   MD146"SERIES MINUS 7- 5"
   MD189"D189 IWER CHECKPOINT"
   MD183M1"WORD RECALL DELAYED - 1"
   MD183M2"WORD RECALL DELAYED - 2"
   MD183M3"WORD RECALL DELAYED - 3"
   MD183M4"WORD RECALL DELAYED - 4"
   MD183M5"WORD RECALL DELAYED - 5"
   MD183M6"WORD RECALL DELAYED - 6"
   MD183M7"WORD RECALL DELAYED - 7"
   MD183M8"WORD RECALL DELAYED - 8"
   MD183M9"WORD RECALL DELAYED - 9"
   MD183M10"WORD RECALL DELAYED - 10"
   MD183M11"WORD RECALL DELAYED - 11"
   MD183M12"WORD RECALL DELAYED - 12"
   MD183M13"WORD RECALL DELAYED - 13"
   MD183M14"WORD RECALL DELAYED - 14"
   MD184"NUMBER GOOD - DELAYED"
   MD185"NUMBER WRONG - DELAYED"
   MD186"NUMBER FORGOTTEN - DELAYED"
   MD187"NONE REMEMBERED - DELAYED"
   MD150"COGNITION INTRO"
   MD151"TODAYS DATE- MONTH"
   MD152"TODAYS DATE- DAY"
   MD153"TODAYS DATE- YEAR"
   MD154"TODAYS DATE- DAY OF WEEK"
   MD155"TOOL USED TO CUT PAPER"
   MD156"NAME OF PRICKLY DESERT PLANT"
   MD157"WHO IS THE PRESIDENT OF US"
   MD158"WHO IS THE VICE-PRESIDENT OF US"
   MD159"D159 CONTINUE IW- VOCAB WORDS"
   MD161"MEANING OF REPAIR/CONCEAL"
   MD163"MEANING OF FABRIC/ENORMOUS"
   MD165"MEANING OF DOMESTIC/PERIMETER"
   MD167"MEANING OF REMORSE/COMPASSION"
   MD169"MEANING OF PLAGIARIZE/AUDACIOUS"
   MD178"CHANCE GET DISEASE"
   MD179"LOTTERY SPLIT 5 WAYS"
   MD180"INTEREST ON SAVINGS"
   MD194"INTRO TO ANIMALS"
   MD196"TOTAL ANIMALS ANSWERS"
   MD197"ANIMAL MISTAKES"
   MD198"ANIMAL MISTAKES NUMBER"
   MD199"TIMING TOOL USED"
   MD200"INTRO-QUANTITATIVE NUMBER SERIES"
   MD240"INTRO-QUANTITATIVE NUMBER SERIES"
   MD241"INTRO-QUANTITATIVE NUMBER SERIES"
   MD242"INTRO-QUANTITATIVE NUMBER SERIES"
   MD201"NUMBER SERIES-G1"
   MD202"NUMBER SERIES-H1"
   MD203"NUMBER SERIES-I1"
   MD204"NUMBER SERIES-A1"
   MD205"NUMBER SERIES-B1"
   MD206"NUMBER SERIES-C1"
   MD207"NUMBER SERIES-D1"
   MD208"NUMBER SERIES-E1"
   MD209"NUMBER SERIES-F1"
   MD210"NUMBER SERIES-J1"
   MD211"NUMBER SERIES-K1"
   MD212"NUMBER SERIES-L1"
   MD213"NUMBER SERIES-M1"
   MD214"NUMBER SERIES-N1"
   MD215"NUMBER SERIES-O1"
   MD221"NUMBER SERIES-G2"
   MD222"NUMBER SERIES-H2"
   MD223"NUMBER SERIES-I2"
   MD224"NUMBER SERIES-A2"
   MD225"NUMBER SERIES-B2"
   MD226"NUMBER SERIES-C2"
   MD227"NUMBER SERIES-D2"
   MD228"NUMBER SERIES-E2"
   MD229"NUMBER SERIES-F2"
   MD230"NUMBER SERIES-J2"
   MD231"NUMBER SERIES-K2"
   MD232"NUMBER SERIES-L2"
   MD233"NUMBER SERIES-M2"
   MD234"NUMBER SERIES-N2"
   MD235"NUMBER SERIES-O2"
   MNSSCORE"CALCULATED NUMBER SERIES SCORE"
   MNSSCORESE"STANDARD ERROR OF NUMBER SERIES SCORE"
   MD216"NS CALC 1-3"
   MD217"NS TOTAL DK/RF"
   MD170"TICS SCORE COUNT"
   MD172"D172 FLAG ASSIST - D"
   MD171"ASSIST SECTION D - COGNITIVE"
   MD501"RATE MEMORY- PC"
   MD502"COMPARE MEM TO PREV WAVE- PC"
   MD505"MEM/INTELLIGENCE INTRO- P C"
   MD506"RATE R AT REMEMBERING THINGS- PC"
   MD507"ORGANIZATION IMPROVED- PC"
   MD508"ORGANIZATION WORSE- PC"
   MD509"RATE R AT REMEMBERING RECENT EVENTS- PC"
   MD510"REMEMBERING RECENT EVENTS IMPROVED- PC"
   MD511"REMEMBERING RECENT EVENTS WORSE- PC"
   MD512"RATE R AT CONVERSATION RECALL- PC"
   MD513"CONVERSATION RECALL IMPROVED- PC"
   MD514"CONVERSATION RECALL WORSE- PC"
   MD515"RATE REMEMBERING OWN PHONE NUM- PC"
   MD516"REMEMBERING OWN PHONE NUM IMPROVE- PC"
   MD517"REMEMBERING OWN PHONE NUM WORSE- PC"
   MD518"RATE REMEMBERING CURRENT DY/MO- PC"
   MD519"REMEMBERING CURRENT DY/MO IMPROVE- PC"
   MD520"REMEMBERING CURRENT DY/MO WORSE- PC"
   MD521"RATE REMEMBERING WHERE THINGS KEPT- PC"
   MD522"WHERE THINGS ARE KEPT IMPROVED- PC"
   MD523"WHERE THINGS ARE KEPT WORSE- PC"
   MD524"RATE FINDING THINGS IN DIFF PLACES- PC"
   MD525"FINDING THINGS IMPROVED- PC"
   MD526"FINDING THINGS WORSE- PC"
   MD527"RATE WORKING WITH FAMILIAR MACHINES- PC"
   MD528"WORKING WITH FAMILIAR MACHINES IMPR- PC"
   MD529"WORKING WITH FAMILIAR MACHINES WORSE- PC"
   MD530"RATE LEARNING NEW MACHINES- PC"
   MD531"LEARNING NEW MACHINES IMPROVED- PC"
   MD532"LEARNING NEW MACHINES WORSE- PC"
   MD533"RATE LEARNING NEW THINGS IN GENERAL- PC"
   MD534"LEARNING ABILITY IMPROVE- PC"
   MD535"LEARNING ABILITY WORSE- PC"
   MD536"RATE ABILITY TO FOLLOW STORY- PC"
   MD537"ABILITY TO FOLLOW STORY IMPROVE- PC"
   MD538"ABILITY TO FOLLOW STORY WORSE- PC"
   MD539"RATE MAKING DECISIONS- PC"
   MD540"MAKE DECISIONS IMPROVE- PC"
   MD541"MAKE DECISIONS WORSE- PC"
   MD542"RATE HANDLING SHOPPING MONEY- PC"
   MD543"HANDLING SHOPPING MONEY IMPROVE- PC"
   MD544"HANDLING SHOPPING MONEY WORSE- PC"
   MD545"RATE HANDLING FINANCES- PC"
   MD546"HANDLING FINANCES IMPROVE- PC"
   MD547"HANDLING FINANCES WORSE- PC"
   MD548"RATE HANDLING DAILY ARITHMETIC PROBS- PC"
   MD549"HANDLING ARITHMETIC PROBLEMS IMPROVE- PC"
   MD550"HANDLING  ARITHMETIC PROBLEMS WORSE- PC"
   MD551"RATE REASONING- PC"
   MD552"REASONING IMPROVE- PC"
   MD553"REASONING WORSE- PC"
   MD554"GET LOST IN FAMILIAR PLACES- PC"
   MD555"WANDER OFF- PC"
   MD556"CAN R BE LEFT ALONE- PC"
   MD557"DOES R HALLUCINATE- PC"
   MVDATE"2010 DATA MODEL VERSION"
   MVERSION"2010 DATA RELEASE VERSION"
.
execute.
save  /outfile 'c:\hrs2010\spss\H10D_R.sav'.
execute.
