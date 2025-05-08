void compilePdfs()
{
	cout<<"===========	First Line 		================"<<endl;
	gSystem->AddIncludePath("-I$ROOFITSYS/include");
	cout<<"===========	processing RooCMSShape.cc	========"<<endl;
	gROOT->ProcessLine(".L /depot/cms/private/users/shar1172/copperheadV2_MergeFW/src/lib/ebeMassResCalibration/PDFs/RooCMSShape.cc+");
	// cout<<"===========	processing PdfDiagonalizer.cc	========"<<endl;
	// gROOT->ProcessLine(".L PdfDiagonalizer.cc+");
	// cout<<"===========	processing Util.cxx		========"<<endl;
	// gROOT->ProcessLine(".L Util.cxx+");
}
