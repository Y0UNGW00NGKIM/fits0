#include "TRandom2.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TGClient.h"
#include "TStyle.h"
#include "TMatrixD.h"
#include "TDecompChol.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TROOT.h"


#include <iostream>
using namespace std;

using TMath::Log;

//parms
const double xmin=1;
const double xmax=20;
const int npoints=12;
const double sigma=0.2;

double f(double x){
  const double a=0.5;
  const double b=1.3;
  const double c=0.5;
  return a+b*Log(x)+c*Log(x)*Log(x);
}

void getX(double *x){
  double step=(xmax-xmin)/npoints;
  for (int i=0; i<npoints; i++){
    x[i]=xmin+i*step;
  }
}

void getY(const double *x, double *y, double *ey){
  static TRandom2 tr(0);
  for (int i=0; i<npoints; i++){
    y[i]=f(x[i])+tr.Gaus(0,sigma);
    ey[i]=sigma;
  }
}


void leastsq(){
  double x[npoints];
  double y[npoints];
  double ey[npoints];
  getX(x);
  getY(x,y,ey);
  auto tg = new TGraphErrors(npoints,x,y,0,ey);
  tg->Draw("alp");
}

int main(int argc, char **argv){
  TApplication theApp("App", &argc, argv); // init ROOT App for displays
    gROOT -> SetBatch(kTRUE);

  // ******************************************************************************
  // ** this block is useful for supporting both high and std resolution screens **
  UInt_t dh = gClient->GetDisplayHeight()/2;   // fix plot to 1/2 screen height  
  //UInt_t dw = gClient->GetDisplayWidth();
  if (dh==0) dh = 600; else dh = dh/2;
  UInt_t dw = (UInt_t)(1.1*dh);
  // ******************************************************************************

  gStyle->SetOptStat(0); // turn off histogram stats box


  TCanvas *tc = new TCanvas("c1","Sample dataset",dw,dh);

  double lx[npoints];
  double ly[npoints];
  double ley[npoints];

  getX(lx);
  getY(lx,ly,ley);
  auto tgl = new TGraphErrors(npoints,lx,ly,0,ley);
  tgl->SetTitle("Pseudoexperiment;x;y");
  
  // An example of one pseudo experiment
  tgl->Draw("alp");
  tc->Draw();


  
  // *** modify and add your code here ***


  TMatrixD A1(npoints,3);
  TMatrixD W1(npoints,npoints); W1.Zero();
  for (int i=0;i<npoints;i++){
    double ln=Log(lx[i]);
    A1(i,0)=1.0; A1(i,1)=ln; A1(i,2)=ln*ln;
    W1(i,i)=1.0/(sigma*sigma);
  }
  TMatrixD AT1(TMatrixD::kTransposed,A1);
  TMatrixD N1 = AT1*W1*A1;
  TMatrixD Ninv1(N1);
  TDecompChol chol1(N1);
  Ninv1 = N1; Ninv1.Invert();
  TVectorD yv1(npoints);
  for (int i=0;i<npoints;i++) yv1[i]=ly[i];
  TVectorD beta1 = Ninv1 * (AT1*(W1*yv1));

  const int ngrid=400;
  auto tgfit = new TGraph(ngrid);
  for (int i=0;i<ngrid;i++){
    double xx = xmin + (xmax-xmin)*i/(ngrid-1);
    double ln = Log(xx);
    double yy = beta1[0] + beta1[1]*ln + beta1[2]*ln*ln;
    tgfit->SetPoint(i,xx,yy);
  }
  tgfit->SetLineColor(kRed+1);
  tgfit->Draw("L SAME");

  auto leg = new TLegend(0.12,0.78,0.60,0.92);
  leg->AddEntry(tgl,"data","lep");
  leg->AddEntry(tgfit,Form("fit: a=%.3f, b=%.3f, c=%.3f",beta1[0],beta1[1],beta1[2]),"l");
  leg->Draw();

  tc->SaveAs("data_fit_cpp.png");

  TH2F *h1 = new TH2F("h1","Parameter b vs a;a;b",60,-0.5,1.5,60,0.3,2.3);
  TH2F *h2 = new TH2F("h2","Parameter c vs a;a;c",60,-0.5,1.5,60,-0.5,1.5);
  TH2F *h3 = new TH2F("h3","Parameter c vs b;b;c",60,0.3,2.3,60,-0.5,1.5);
  TH1F *h4 = new TH1F("h4","Reduced #chi^{2};#chi^{2}_{#nu};frequency",40,0,3.0);

  // perform many least squares fits on different pseudo experiments here
  // fill histograms w/ required data
  
  const int ntrials=1000;
  TRandom2 tr(12345);

  TMatrixD A(npoints,3);
  TMatrixD W(npoints,npoints); W.Zero();
  for (int i=0;i<npoints;i++){
    double ln=TMath::Log(lx[i]);
    A(i,0)=1.0; A(i,1)=ln; A(i,2)=ln*ln;
    W(i,i)=1.0/(sigma*sigma);
  }
  TMatrixD AT(TMatrixD::kTransposed,A);
  TMatrixD N = AT*W*A;
  TMatrixD Ninv(N);
  TDecompChol chol(N);
  Ninv = N; Ninv.Invert();
  TVectorD yv(npoints);

  for (int t=0; t<ntrials; t++){
    for (int i=0;i<npoints;i++){
      ly[i]=f(lx[i])+tr.Gaus(0,sigma);
      ley[i]=sigma;
      yv[i]=ly[i];
    }
    TVectorD beta = Ninv * (AT*(W*yv));

    double chi2=0.0;
    for (int i=0;i<npoints;i++){
      double ln=TMath::Log(lx[i]);
      double yhat = beta[0] + beta[1]*ln + beta[2]*ln*ln;
      double r = (ly[i]-yhat)/sigma;
      chi2 += r*r;
    }
    double chi2red = chi2/(npoints-3);

    h1->Fill(beta[0],beta[1]);
    h2->Fill(beta[0],beta[2]);
    h3->Fill(beta[1],beta[2]);
    h4->Fill(chi2red);
  }

  TCanvas *tc2 = new TCanvas("c2","my study results",dw,dh);
  tc2->Divide(2,2);
  tc2->cd(1); h1->Draw("colz");
  tc2->cd(2); h2->Draw("colz");
  tc2->cd(3); h3->Draw("colz");
  tc2->cd(4); h4->Draw();
  tc2->SaveAs("study_panels_cpp.png");
  tc2->Draw();

  // **************************************
  
  cout << "Press ^c to exit" << endl;
  theApp.SetIdleTimer(30,".q");  // set up a failsafe timer to end the program  
  theApp.Run();
}