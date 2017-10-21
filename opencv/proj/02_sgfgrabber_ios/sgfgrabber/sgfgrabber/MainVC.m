//
//  MainVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//

#import "MainVC.h"

//==========================
@interface MainVC ()
@property FrameExtractor *frameExtractor;
@end

//=========================
@implementation MainVC

- (void)viewDidLoad {
    [super viewDidLoad];
    self.frameExtractor = [FrameExtractor new];
    self.frameExtractor.delegate = self;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}

#pragma mark - FrameExtractorDelegate protocol
- (void)captured:(UIImage *)image
{
    NSLog(@"Got a frame");
}

@end
