//
//  MainVC.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//


#import "Common.h"
#import "FrameExtractor.h"

@interface MainVC : UIViewController <FrameExtractorDelegate>
// Label for debug output
@property UILabel *lbDbg;
// Callbacks
- (void) mnuAddTestCase;
- (void) mnuEditTestCases;
@end
