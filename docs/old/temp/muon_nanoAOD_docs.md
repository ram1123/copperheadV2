```bash
Muon_bsConstrainedChi2	Float_t	chi2 of beamspot constraint
Muon_bsConstrainedPt	Float_t	pT with beamspot constraint
Muon_bsConstrainedPtErr	Float_t	pT error with beamspot constraint
Muon_charge	Int_t	electric charge
Muon_dxy	Float_t	dxy (with sign) wrt first PV, in cm
Muon_dxyErr	Float_t	dxy uncertainty, in cm
Muon_dxybs	Float_t	dxy (with sign) wrt the beam spot, in cm
Muon_dz	Float_t	dz (with sign) wrt first PV, in cm
Muon_dzErr	Float_t	dz uncertainty, in cm
Muon_eta	Float_t	eta
Muon_highPurity	Bool_t	inner track is high purity
Muon_inTimeMuon	Bool_t	inTimeMuon ID
Muon_ip3d	Float_t	3D impact parameter wrt first PV, in cm
Muon_isGlobal	Bool_t	muon is global muon
Muon_isPFcand	Bool_t	muon is PF candidate
Muon_isStandalone	Bool_t	muon is a standalone muon
Muon_isTracker	Bool_t	muon is tracker muon
Muon_jetIdx	Short_t(index to Jet)	index of the associated jet (-1 if none)
Muon_jetNDauCharged	UChar_t	number of charged daughters of the closest jet
Muon_jetPtRelv2	Float_t	Relative momentum of the lepton with respect to the closest jet after subtracting the lepton
Muon_jetRelIso	Float_t	Relative isolation in matched jet (1/ptRatio-1, pfRelIso04_all if no matched jet)
Muon_looseId	Bool_t	muon is loose muon
Muon_mass	Float_t	mass
Muon_mediumId	Bool_t	cut-based ID, medium WP
Muon_mediumPromptId	Bool_t	cut-based ID, medium prompt WP
Muon_miniIsoId	UChar_t	MiniIso ID from miniAOD selector (1=MiniIsoLoose, 2=MiniIsoMedium, 3=MiniIsoTight, 4=MiniIsoVeryTight)
Muon_miniPFRelIso_all	Float_t	mini PF relative isolation, total (with scaled rho*EA PU corrections)
Muon_miniPFRelIso_chg	Float_t	mini PF relative isolation, charged component
Muon_multiIsoId	UChar_t	MultiIsoId from miniAOD selector (1=MultiIsoLoose, 2=MultiIsoMedium)
Muon_nStations	UChar_t	number of matched stations with default arbitration (segment & track)
Muon_nTrackerLayers	UChar_t	number of layers in the tracker
Muon_pdgId	Int_t	PDG code assigned by the event reconstruction (not by MC truth)
Muon_pfIsoId	UChar_t	PFIso ID from miniAOD selector (1=PFIsoVeryLoose, 2=PFIsoLoose, 3=PFIsoMedium, 4=PFIsoTight, 5=PFIsoVeryTight, 6=PFIsoVeryVeryTight)
Muon_pfRelIso03_all	Float_t	PF relative isolation dR=0.3, total (deltaBeta corrections)
Muon_pfRelIso03_chg	Float_t	PF relative isolation dR=0.3, charged component
Muon_pfRelIso04_all	Float_t	PF relative isolation dR=0.4, total (deltaBeta corrections)
Muon_phi	Float_t	phi
Muon_pt	Float_t	pt
Muon_ptErr	Float_t	ptError of the muon track
Muon_puppiIsoId	UChar_t	PuppiIsoId from miniAOD selector (1=Loose, 2=Medium, 3=Tight)
Muon_segmentComp	Float_t	muon segment compatibility
Muon_sip3d	Float_t	3D impact parameter significance wrt first PV
Muon_svIdx	Short_t(index to Sv)	index of matching secondary vertex
Muon_tightCharge	UChar_t	Tight charge criterion using pterr/pt of muonBestTrack (0:fail, 2:pass)
Muon_tkIsoId	UChar_t	TkIso ID (1=TkIsoLoose, 2=TkIsoTight)
Muon_tkRelIso	Float_t	Tracker-based relative isolation dR=0.3 for highPt, trkIso/tunePpt
nMuon	Int_t	slimmedMuons after basic selection (pt > 15 || (pt > 3 && (passed('CutBasedIdLoose') || passed('SoftCutBasedId') || passed('SoftMvaId') || passed('CutBasedIdGlobalHighPt') || passed('CutBasedIdTrkHighPt'))))
```
