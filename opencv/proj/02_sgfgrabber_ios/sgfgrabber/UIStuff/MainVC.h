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
// The frame extractor dealing with all things video
@property FrameExtractor *frameExtractor;

// Label for various information
@property UILabel *lbBottom;
// Slider for test purposes
//@property UISlider *sldDbg;


//// Callbacks
////============
//- (void) mnuAddTestCase;
//- (void) mnuEditTestCases;

// Other
//=======
// Redraw main screen
- (void) doLayout;
// Debugging helper, shows individual processing stages
- (void) debugFlow:(bool)reset;

@end
