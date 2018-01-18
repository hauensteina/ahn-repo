//
//  MainVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//

// Don't change the order of these two,
// and don't move them down
#import "Ocv.hpp"
#import <opencv2/imgcodecs/ios.h>

#import "MainVC.h"
#import "UIViewController+LGSideMenuController.h"

#import "Globals.h"
#import "Helpers.hpp"
#import "CppInterface.h"

#define DDEBUG

//==========================
@interface MainVC ()
@property FrameExtractor *frameExtractor;
@property UIImageView *cameraView;
// Entry point to app core functionality.
@property CppInterface *cppInterface;
// Data
@property UIImage *img; // The current image
// History of frames. The one at the button press is often shaky.
@property std::vector<cv::Mat> imgQ;

// Buttons etc
@property UIButton *btnGo;
@property UISlider *sldDbg;
@property UISwitch *swiDbg;

// State
@property BOOL frame_grabber_on; // Set to NO to stop the frame grabber
@property BOOL debug_mode;
@property int sliderstate;

@end

//=========================
@implementation MainVC

//----------------
- (id)init
{
    self = [super init];
    if (self) {
        self.title = @"SgfGrabber";
        self.view.backgroundColor =  [UIColor colorWithRed:0.8 green:0.8 blue:0.8 alpha:0.95];
        
        
        self.navigationItem.leftBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:@"Menu"
                                                                                 style:UIBarButtonItemStylePlain
                                                                                target:self
                                                                                action:@selector(showLeftView)];        
    }
    return self;
} // init()

//----------------------
- (void)viewDidLoad
{
    [super viewDidLoad];
    self.frameExtractor = [FrameExtractor new];
    self.cppInterface = [CppInterface new];
    self.frameExtractor.delegate = self;
    self.frame_grabber_on = YES;
    self.sliderstate = 0;
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
    //v.backgroundColor = RGB(0xF3DCA5);
    v.backgroundColor = BLACK;

    // Camera View
    self.cameraView = [UIImageView new];
    self.cameraView.contentMode = UIViewContentModeScaleAspectFit;
    [v addSubview:self.cameraView];
    
    // Buttons etc
    //================
    self.btnGo = [self addButtonWithTitle:@"Go" callback:@selector(btnGo:)];
    
    // Toggle debug mode
    UISwitch *swi = [UISwitch new];
    [swi setOn:NO]; _debug_mode = false;
    //[swi setOn:NO];
    [swi addTarget:self action:@selector(swiDbg:) forControlEvents:UIControlEventValueChanged];
    [v addSubview:swi];
    self.swiDbg = swi;
    
    // Label for various debug info
    UILabel *l = [UILabel new];
    l.hidden = false;
    l.text = @"";
    l.backgroundColor = WHITE;
    [v addSubview:l];
    self.lbDbg = l;
    
    // Debug slider
    UISlider *s = [UISlider new];
    self.sldDbg = s;
    s.minimumValue = 0;
    s.maximumValue = 16;
    [s addTarget:self action:@selector(sldDbg:) forControlEvents:UIControlEventValueChanged];
    s.backgroundColor = RGB (0xf0f0f0);
    [v addSubview:s];
    self.sldDbg.hidden = false;
}

//----------------------------------------------------------------------
- (void) viewWillAppear:(BOOL) animated
{
    [super viewWillAppear: animated];
    [self doLayout];
    
}

//-------------------------------
- (BOOL)prefersStatusBarHidden
{
    return YES;
}

#pragma mark Layout

// Put UI elements into the right place
//----------------------------------------------------------------------
- (void) doLayout
{
    UIView *v = self.view;
    //UIFont *fnt; //UILabel *lab;
    float W = SCREEN_WIDTH;
    float H = SCREEN_HEIGHT;
    //float sth = STATUSBAR_HEIGHT + 10;
    float mh = 55;
    float delta_y = mh * 1.1;
    float y = H - delta_y;
    float lmarg = W / 20;
    float rmarg = W / 20;
    //float labwidth = W - lmarg - rmarg;

    self.cameraView.frame = v.bounds;
    self.cameraView.hidden = NO;
    //[self.view bringSubviewToFront:self.cameraView];

    // Button
    self.btnGo.frame = CGRectMake (lmarg, y, W /5 , mh);
    self.btnGo.titleLabel.font = [UIFont fontWithName:@"HelveticaNeue" size: 40];
    [self.btnGo setTitleColor:DARKRED forState:UIControlStateNormal];
    // Debug switch
    self.swiDbg.frame = CGRectMake (lmarg + W/5 + W/10, y + mh/4, W /5 , mh);
    // Debug label
    int left = lmarg + W/5 + W/10 + W/5;
    int width = W-rmarg-left;
    self.lbDbg.frame = CGRectMake (left, y, width , mh);
    // Debug slider
    y -= delta_y;
    self.sldDbg.frame = CGRectMake(lmarg, y, W - lmarg - rmarg, mh);

} // doLayout
    

//---------------------------------------------------
- (UIButton*) addButtonWithTitle: (NSString *) title
                        callback: (SEL) callback
{
    UIView *parent = self.view;
    id target = self;
    
    UIButton *b = [[UIButton alloc] init];
    [b.layer setBorderWidth:1.0];
    [b.layer setBorderColor:[RGB (0x202020) CGColor]];
    //b.titleLabel.font =[UIFont fontWithName:@"HelveticaNeue" size: 10];
    b.backgroundColor = RGB (0xf0f0f0);
    b.frame = CGRectMake(0, 0, 72, 44);
    [b setTitle: title forState: UIControlStateNormal];
    [b setTitleColor: WHITE forState: UIControlStateNormal];
    //b.titleLabel.baselineAdjustment = UIBaselineAdjustmentAlignCenters;
    [b addTarget:target action:callback forControlEvents: UIControlEventTouchUpInside];
    [parent addSubview: b];
    //b.contentVerticalAlignment = UIControlContentVerticalAlignmentFill;
    return b;
} // addButtonWithTitle

#pragma mark - Button etc callbacks

// Slider for Debugging
//-----------------------------------
- (void) sldDbg:(id) sender
{
    int tt = [self.sldDbg value];
    self.cppInterface.sldDbg = tt;
    self.lbDbg.text = nsprintf( @"%d", tt);
    //_sliderstate = 0;
}

// Debug on/off
//----------------------------
- (void) swiDbg:(id) sender
{
    self.debug_mode = [self.swiDbg isOn];
}

// Debugging helper, shows individual processing stages
//------------------------------------------------------
- (void) btnGo: (id) sender
{
    if (self.debug_mode) {
        UIImage *img;
        while(1) {
            switch (_sliderstate) {
                case 0:
                    _sliderstate++;
                    //_sliderstate=100;
                    self.frame_grabber_on = NO;
                    [self.frameExtractor suspend];
                    img = [self.cppInterface f00_blobs:_imgQ];
                    [self.cameraView setImage:img];
                    break;
                case 1:
                    img = [self.cppInterface f01_vert_lines];
                    if (!img) { _sliderstate=2; continue; }
                    [self.cameraView setImage:img];
                    break;
                case 2:
                    img = [self.cppInterface f02_horiz_lines];
                    if (!img) { _sliderstate=3; continue; }
                    [self.cameraView setImage:img];
                    break;
                case 3:
                    _sliderstate++;
                    img = [self.cppInterface f03_corners];
                    [self.cameraView setImage:img];
                    break;
                case 4:
                    _sliderstate++;
                    img = [self.cppInterface f04_zoom_in];
                    [self.cameraView setImage:img];
                    break;
                case 5:
                    _sliderstate++;
                    img = [self.cppInterface f05_dark_places];
                    [self.cameraView setImage:img];
                    break;
                case 6:
                    _sliderstate++;
                    img = [self.cppInterface f06_mask_dark];
                    [self.cameraView setImage:img];
                    break;
                case 7:
                    _sliderstate++;
                    img = [self.cppInterface f07_white_holes];
                    [self.cameraView setImage:img];
                    break;
                case 8:
                    _sliderstate++; continue; // skip;
                    img = [self.cppInterface f08_features];
                    if (!img) { _sliderstate=9; continue; }
                    [self.cameraView setImage:img];
                    break;
                case 9:
                    _sliderstate++;
                    img = [self.cppInterface f09_classify];
                    [self.cameraView setImage:img];
                    break;
                default:
                    _sliderstate=0;
                    self.frame_grabber_on = YES;
                    [self.frameExtractor resume];
                    g_app.mainVC.lbDbg.text = @"";
            } // switch
            break;
        } // while(1)
    } // if
} // btnGo()

#pragma mark - FrameExtractorDelegate protocol
//-----------------------------------------------
- (void)captured:(UIImage *)image
{
    //self.cameraView.hidden = NO;
    //static int i = 0;
    if (self.frame_grabber_on) {
        //i++;
        //PLOG("frame:%d\n",i);
        if (self.debug_mode) {
            [self.cameraView setImage:image];
            _img = image;
            cv::Mat m;
            UIImageToMat( _img, m);
            resize( m, m, 350);
            cv::cvtColor( m, m, CV_RGBA2RGB); // Yes, RGBA not BGR
            ringpush( _imgQ , m, 4); // keep 4 frames
        }
        else {
            self.frame_grabber_on = NO;
            [self.frameExtractor suspend];
            UIImage *processedImg = [self.cppInterface real_time_flow:image];
            self.img = processedImg;
            [self.cameraView setImage:self.img];
            self.frame_grabber_on = YES;
            [self.frameExtractor resume];
        }
    }
} // captured()

#pragma mark LGSideMenuController Callbacks

//---------------------
- (void)showLeftView
{
    [self.sideMenuController showLeftViewAnimated:YES completionHandler:nil];
}

//------------------------
- (void)showRightView
{
    [self.sideMenuController showRightViewAnimated:YES completionHandler:nil];
}

// Save current image and board position as jpg and sgf.
// Filenames are testcase_nnnnn.jpg|sgf.
// The new nnnnn is one higher than the largest one found in the
// file systm.
//---------------------------
- (void) mnuSaveAsTestCase
{
    const int BUFSZ = 1000;
    char buf[BUFSZ+1];
    std::string docpath = [getFullPath( @"/") UTF8String];
    
    // Find next file name
    std::vector<cv::String> fnames;
    cv::glob( docpath + "/" + TESTCASE_PREFIX + "*.jpg", fnames);
    int fnum = 0;
    if (SZ(fnames)) {
        vec_sort( fnames);
        std::string last = fnames.back();
        std::vector<std::string> parts;
        str_split( last, parts, '_');
        int num = std::stoi( parts.back());
        fnum = num + 1;
    }
    std::snprintf( buf, BUFSZ, TESTCASE_PREFIX "%05d", fnum);

    // Save image
    std::string fname = docpath + "/" + buf + ".jpg";
    cv::Mat m;
    cv::cvtColor( _cppInterface.small_img, m, CV_RGB2BGR);
    cv::imwrite( fname, m);
    
    // Save SGF
    auto sgf = generate_sgf( "Testcase " + std::to_string( fnum) , _cppInterface.diagram);
    fname = docpath + "/" + buf + ".sgf";
    std::ofstream ofs;
    ofs.open( fname);
    ofs << sgf;
    ofs.close();
    
    popup( @"Image added as Test Case", @"");
} // mnuSaveAsTestCase()

// Show test cases from filesystem in a tableview, pick one.
//-------------------------------------------------------------
- (void) mnuEditTestCases
{
    [g_app.navVC pushViewController:g_app.editTestCaseVC animated:YES];
} // mnuSetCurrentTestCase()

@end
