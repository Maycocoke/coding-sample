
clear all

global HOMEdir ../../
global RAWDATAdir ${HOMEdir}raw_data/
global INTdir ${HOMEdir}intermediate_data/
global PROCdir ${HOMEdir}processed_data/
global DOdir ${HOMEdir}dofiles/
global TABLESdir ${HOMEdir}tables_and_figures/

run ${DOdir}/neisser/analysis_progs.do

****** What are supposed to appear******
/*destring owth owthstd , replace

gen prec_owth = 1/owthstd^2

reshape long SOMEVARIABLES, i(country surveyyear TNumber) j(elasttype)

reg owth confidence_government i.TNumber [aw=prec_owth] if elasttype==SOMETYPE , robust*/

********************************************************************************
*** Generation of combined table 
********************************************************************************

local strategy wls 
*local option cluster(g_id)
local weight1 [aweight = precision_sq]
local weight2 [aweight = precision]

cap prog drop renameestimates
prog renameestimates , eclass
	syntax anything , newname(name) [modlocal(name) newlocalname(name) all]
	mat eb = e(b)
	loc cn: colnames eb
	loc cn: subinstr loc cn "`anything'" "`newname'" , `all'
	mat colnames eb = `cn'
	ereturn repost b=eb , rename
	if "`modlocal'"!="" {
		ereturn local `newlocalname' `e(`modlocal')'
	}
end


/*
gen q1num =cond(index(q1total,"–")>0,-real(substr(q1total,4,.)),real(q1total))*/


******************************
***weight[aweight = precision_sq]
******************************
foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	encode elasttype, generate(elasttypenum)
	reg value `v' i.TNumber##i.elasttypenum `weight1', cluster(estimmodel)
	renameestimates `v' , newname(taxpref1)
	eststo , prefix(barmain1)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	reg value `v' i.TNumber `weight1' if elasttype=="owth", cluster(estimmodel)
	renameestimates `v' , newname(taxpref2)
	eststo , prefix(barmain2)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	reg value `v' i.TNumber `weight1' if elasttype=="owthc", cluster(estimmodel)
	renameestimates `v' , newname(taxpref3)
	eststo , prefix(barmain3)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	reg value `v' i.TNumber `weight1' if elasttype=="imh", cluster(estimmodel)
	renameestimates `v' , newname(taxpref4)
	eststo , prefix(barmain4)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	reg value `v' i.TNumber `weight1' if elasttype=="emh", cluster(estimmodel)
	renameestimates `v' , newname(taxpref5)
	eststo , prefix(barmain5)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	reg value `v' i.TNumber `weight1' if elasttype=="owemp", cluster(estimmodel)
	renameestimates `v' , newname(taxpref6)
	eststo , prefix(barmain6)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	encode elasttype, generate(elasttypenum)
	reg value `v' i.TNumber##i.elasttypenum `weight1', cluster(estimmodel)
	renameestimates `v' , newname(taxpref7)
	eststo , prefix(barmain7)
}

erase "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex"
esttab barmain1* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex" ///
	, ///
	replace fragment booktabs se nogaps nolines ///
	keep(taxpref1) ///Only includes the variable taxpref1 in the table.
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	stats(N, ///Includes additional statistics (number of observations and number of clusters) 
		labels("\addlinespace Observations") fmt(%9.0fc) /// with specified formatting.
	) ///
	eqlabels(none) nomtitle ///
	prehead("& \multicolumn{7}{c}{Dependent variable: ETI. Government-related social preferences proxy is:} \\ \cmidrule(lr){2-8} & Confidence in government & Confidence in political parties & Confidence in parliament & Confidence in civil service & Income should be made more equal & Gvt should increase ownership of businesses & Proud to be a citizen \\ ") ///before table: information about the dependent variable and column headers.
	posthead("\midrule \addlinespace \multicolumn{8}{c}{\textit{Panel A: Main estimates}} \\ \addlinespace") /// after the table header but before the body. It includes information about the panel and adds space.
	coeflabel( ///provides labels for the coefficients.
		taxpref1 "Gvt-related social preferences proxy" ///
	) /// 
//	

esttab barmain2* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref2) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref2 "1 Total hours" ///
	) /// 
//

esttab barmain3* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref3) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref3 "2 Total hours compensated" ///
	) /// 
//

esttab barmain4* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref4) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref4 "3 Intensive margin hour" ///
	) /// 
//

esttab barmain5* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref5) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref5 "4 Extensive margin hour" ///
	) /// 
//

esttab barmain6* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref6) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref6 "5 Extensive margin participation" ///
	) /// 
//

esttab barmain7* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main1.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref7) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref7 "6 All the above in one regression" ///
	) /// 
//


******************************
***weight[aweight = precision]
******************************
foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision = 1/std
	encode elasttype, generate(elasttypenum)
	*drop if country==376
	reg value `v' i.TNumber##i.elasttypenum `weight2', cluster(estimmodel)
	renameestimates `v' , newname(taxpref1)
	eststo , prefix(barmaintwo1)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision = 1/std
	*drop if country==376
	reg value `v' i.TNumber `weight2' if elasttype=="owth", cluster(estimmodel)
	renameestimates `v' , newname(taxpref2)
	eststo , prefix(barmaintwo2)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision = 1/std
	*drop if country==376
	reg value `v' i.TNumber `weight2' if elasttype=="owthc", cluster(estimmodel)
	renameestimates `v' , newname(taxpref3)
	eststo , prefix(barmaintwo3)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision= 1/std
	*drop if country==376
	reg value `v' i.TNumber `weight2' if elasttype=="imh", cluster(estimmodel)
	renameestimates `v' , newname(taxpref4)
	eststo , prefix(barmaintwo4)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision = 1/std
	*drop if country==376
	reg value `v' i.TNumber `weight2' if elasttype=="emh", cluster(estimmodel)
	renameestimates `v' , newname(taxpref5)
	eststo , prefix(barmaintwo5)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision= 1/std
	*drop if country==376
	reg value `v' i.TNumber `weight2' if elasttype=="owemp", cluster(estimmodel)
	renameestimates `v' , newname(taxpref6)
	eststo , prefix(barmaintwo6)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision= 1/std
	encode elasttype, generate(elasttypenum)
	*drop if country==376
	reg value `v' i.TNumber##i.elasttypenum `weight2', cluster(estimmodel)
	renameestimates `v' , newname(taxpref7)
	eststo , prefix(barmaintwo7)
}

erase "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex"
esttab barmaintwo1* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex" ///
	, ///
	replace fragment booktabs se nogaps nolines ///
	keep(taxpref1) ///Only includes the variable taxpref1 in the table.
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	stats(N, ///Includes additional statistics (number of observations and number of clusters) 
		labels("\addlinespace Observations") fmt(%9.0fc) /// with specified formatting.
	) ///
	eqlabels(none) nomtitle ///
	prehead("& \multicolumn{7}{c}{Dependent variable: ETI. Government-related social preferences proxy is:} \\ \cmidrule(lr){2-8} & Confidence in government & Confidence in political parties & Confidence in parliament & Confidence in civil service & Income should be made more equal & Gvt should increase ownership of businesses & Proud to be a citizen \\ ") ///before table: information about the dependent variable and column headers.
	posthead("\midrule \addlinespace \multicolumn{8}{c}{\textit{Panel A: Main estimates}} \\ \addlinespace") /// after the table header but before the body. It includes information about the panel and adds space.
	coeflabel( ///provides labels for the coefficients.
		taxpref1 "Gvt-related social preferences proxy" ///
	) /// 
//	

esttab barmaintwo2* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref2) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref2 "1 Total hours" ///
	) /// 
//

esttab barmaintwo3* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref3) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref3 "2 Total hours compensated" ///
	) /// 
//

esttab barmaintwo4* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref4) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref4 "3 Intensive margin hour" ///
	) /// 
//

esttab barmaintwo5* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref5) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref5 "4 Extensive margin hour" ///
	) /// 
//

esttab barmaintwo6* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref6) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref6 "5 Extensive margin participation" ///
	) /// 
//

esttab barmaintwo7* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main2.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref7) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref7 "6 All the above in one regression" ///
	) /// 
//


******************************
***weight[no weight]
******************************
foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	encode elasttype, generate(elasttypenum)
	*drop if country==376
	reg value `v' i.TNumber##i.elasttypenum, cluster(estimmodel)
	renameestimates `v' , newname(taxpref1)
	eststo , prefix(barmainthree1)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	*drop if country==376
	reg value `v' i.TNumber  if elasttype=="owth", cluster(estimmodel)
	renameestimates `v' , newname(taxpref2)
	eststo , prefix(barmainthree2)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	*drop if country==376
	reg value `v' i.TNumber  if elasttype=="owthc", cluster(estimmodel)
	renameestimates `v' , newname(taxpref3)
	eststo , prefix(barmainthree3)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	*drop if country==376
	reg value `v' i.TNumber  if elasttype=="imh", cluster(estimmodel)
	renameestimates `v' , newname(taxpref4)
	eststo , prefix(barmainthree4)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	*drop if country==376
	reg value `v' i.TNumber  if elasttype=="emh", cluster(estimmodel)
	renameestimates `v' , newname(taxpref5)
	eststo , prefix(barmainthree5)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	*drop if country==376
	reg value `v' i.TNumber  if elasttype=="owemp", cluster(estimmodel)
	renameestimates `v' , newname(taxpref6)
	eststo , prefix(barmainthree6)
}

foreach v in confidence_government confidence_politicalparty confidence_parliment confidence_civilservice equalincome economicvalue_ownership proudnation {
	use "${PROCdir}bargainetal2014/bargainetal`v'.dta", clear 
	keep owth owthstd owthc owthcstd imh imhstd emh emhstd owemp owempstd TNumber surveyyear country estimmodel wavecorresponding `v'
	rename (owth owthc imh emh owemp) (valueowth valueowthc valueimh valueemh valueowemp)
	rename (owthstd owthcstd imhstd emhstd owempstd)(stdowth stdowthc stdimh stdemh stdowemp)
    reshape long value std, i(country surveyyear TNumber) j(elasttype) string
    gen precision_sq = 1/std^2
	encode elasttype, generate(elasttypenum)
	*drop if country==376
	reg value `v' i.TNumber##i.elasttypenum , cluster(estimmodel)
	renameestimates `v' , newname(taxpref7)
	eststo , prefix(barmainthree7)
}

erase "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex"
esttab barmainthree1* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex" ///
	, ///
	replace fragment booktabs se nogaps nolines ///
	keep(taxpref1) ///Only includes the variable taxpref1 in the table.
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	stats(N, ///Includes additional statistics (number of observations and number of clusters) 
		labels("\addlinespace Observations") fmt(%9.0fc) /// with specified formatting.
	) ///
	eqlabels(none) nomtitle ///
	prehead("& \multicolumn{7}{c}{Dependent variable: ETI. Government-related social preferences proxy is:} \\ \cmidrule(lr){2-8} & Confidence in government & Confidence in political parties & Confidence in parliament & Confidence in civil service & Income should be made more equal & Gvt should increase ownership of businesses & Proud to be a citizen \\ ") ///before table: information about the dependent variable and column headers.
	posthead("\midrule \addlinespace \multicolumn{8}{c}{\textit{Panel A: Main estimates}} \\ \addlinespace") /// after the table header but before the body. It includes information about the panel and adds space.
	coeflabel( ///provides labels for the coefficients.
		taxpref1 "Gvt-related social preferences proxy" ///
	) /// 
//	

esttab barmainthree2* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref2) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref2 "1 Total hours" ///
	) /// 
//

esttab barmainthree3* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref3) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref3 "2 Total hours compensated" ///
	) /// 
//

esttab barmainthree4* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref4) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref4 "3 Intensive margin hour" ///
	) /// 
//

esttab barmainthree5* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref5) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref5 "4 Extensive margin hour" ///
	) /// 
//

esttab barmainthree6* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref6) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref6 "5 Extensive margin participation" ///
	) /// 
//

esttab barmainthree7* ///
	using "${TABLESdir}/short/tab_metaanalysis_bargainetal_main3.tex" ///
	, ///
	append fragment booktabs se nogaps nolines nonumbers noobs ///
	keep(taxpref7) ///
	star(* .1 ** .05 *** .01) b(a2) se(a2) ///
	eqlabels(none) nomtitle ///
	prehead("") ///
	posthead("") ///
	coeflabel( ///
		taxpref7 "6 All the above in one regression" ///
	) /// 
//
