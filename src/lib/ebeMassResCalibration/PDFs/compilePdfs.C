void compilePdfs()
{
	cout<<"===========	First Line 		================"<<endl;
	gSystem->AddIncludePath("-I$ROOFITSYS/include");
	cout<<"===========	processing RooCMSShape.cc	========"<<endl;
	gROOT->ProcessLine(".L RooCMSShape.cc+");
}
