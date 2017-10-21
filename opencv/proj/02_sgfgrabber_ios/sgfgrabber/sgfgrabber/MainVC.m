//
//  MainVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//

#import "MainVC.h"
#import "GrabFuncs.h"

//==========================
@interface MainVC ()
@property FrameExtractor *frameExtractor;
@property GrabFuncs *grabFuncs;
@property UIImageView *cameraView;
@end

//=========================
@implementation MainVC

//----------------------
- (void)viewDidLoad
{
    [super viewDidLoad];
    self.frameExtractor = [FrameExtractor new];
    self.grabFuncs = [GrabFuncs new];
    self.frameExtractor.delegate = self;
    NSString *tstr = [GrabFuncs opencvVersion];
    int tt=42;
}
//----------------------------------
- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
}

#pragma mark - View Lifecycle
// Allocate all UI elements.
//----------------------------------------------------------------------
- (void) loadView
{
    self.view = [UIView new];
    UIView *v = self.view;
    v.autoresizesSubviews = NO;
    v.opaque = YES;
    v.backgroundColor = RGB(0xF3DCA5);
    
    // Camera View
    self.cameraView = [UIImageView new];
    [v addSubview:self.cameraView];
}

//----------------------------------------------------------------------
- (void) viewWillAppear:(BOOL) animated
{
    [super viewWillAppear: animated];
    UIView *v = self.view;
    self.cameraView.frame = v.bounds;
    self.cameraView.hidden = NO;
    [self.view bringSubviewToFront:self.cameraView];
}

#pragma mark - FrameExtractorDelegate protocol
//---------------------------------
- (void)captured:(UIImage *)image
{
    //self.cameraView.hidden = NO;
    [self.cameraView setImage:image];
}


@end
