//
//  MainVC.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//

#import "Common.h"
#import "CppInterface.h"
#import "FrameExtractor.h"

@interface MainVC : UIViewController <FrameExtractorDelegate>
// Entry point to core app functionality.
@property CppInterface *cppInterface;
// Label for debug output
@property UILabel *lbDbg;
// Slider for various test purposes
@property UISlider *sldDbg;

// Callbacks
//============
- (void) mnuAddTestCase;
- (void) mnuEditTestCases;

// Other
//=======
// Debugging helper, shows individual processing stages
- (void) debugFlow:(bool)reset;

@end
