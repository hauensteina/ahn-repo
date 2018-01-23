//
//  MainVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//

#import "MainVC.h"
#import "UIViewController+LGSideMenuController.h"

#import "Globals.h"
#import "CppInterface.h"

#define DDEBUG

//==========================
@interface MainVC ()
@property FrameExtractor *frameExtractor;
@property UIImageView *cameraView;
// Data
@property UIImage *img; // The current image

// Buttons etc
@property UIButton *btnGo;
@property UISwitch *swiDbg;

@property UIButton *btnCam;
@property UIImage *imgVideoBtn;
@property UIImage *imgPhotoBtn;

// State
@property BOOL frame_grabber_on; // Set to NO to stop the frame grabber
@property BOOL debug_mode;
@property int debugstate;

@end

//=========================
@implementation MainVC

//----------------
- (id)init
{
    self = [super init];
    if (self) {
        self.title = @"SgfGrabber";
        self.view.backgroundColor = BGCOLOR;
        
        
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
    self.debugstate = 0;
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
    
    // Button for video or image
    self.btnCam = [self addButtonWithTitle:@"" callback:@selector(btnCam:)];
    self.imgPhotoBtn = [UIImage imageNamed:@"photo_icon.png"];
    self.imgVideoBtn = [UIImage imageNamed:@"video_icon.png"];
    [self.btnCam setBackgroundImage:self.imgVideoBtn forState:UIControlStateNormal];
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
    self.btnGo.frame = CGRectMake( lmarg, y, W /5 , mh);
    self.btnGo.titleLabel.font = [UIFont fontWithName:@"HelveticaNeue" size: 40];
    [self.btnGo setTitleColor:DARKRED forState:UIControlStateNormal];

    // Camera button
    [self.btnCam setBackgroundImage:self.imgPhotoBtn forState:UIControlStateNormal];
    if ([g_app.menuVC videoMode]) {
        [self.btnCam setBackgroundImage:self.imgVideoBtn forState:UIControlStateNormal];
    }
    int r = 70;
    self.btnCam.frame = CGRectMake( W/2 - r/2, y-150, r , r);
    CALayer *layer = self.btnCam.layer;
    layer.backgroundColor = [[UIColor clearColor] CGColor];
    layer.borderColor = [[UIColor clearColor] CGColor];
    //layer.cornerRadius = 8.0f;
    //layer.borderWidth = 1.0f;

    
    // Debug switch
    self.swiDbg.frame = CGRectMake( lmarg + W/5 + W/10, y + mh/4, W /5 , mh);
    // Debug label
    int left = lmarg + W/5 + W/10 + W/5;
    int width = W-rmarg-left;
    self.lbDbg.frame = CGRectMake( left, y, width , mh);
    // Debug slider
    y -= delta_y;
    self.sldDbg.frame = CGRectMake( lmarg, y, W - lmarg - rmarg, mh);

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

// Tapping on the screen
//----------------------------------------------------------------
- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    [super touchesBegan:touches withEvent:event];
    if ([g_app.menuVC debugMode]) {
        [self debugFlow:false];
    }
}

// Slider for Debugging
//-----------------------------------
- (void) sldDbg:(id) sender
{
    int tt = [self.sldDbg value];
    self.lbDbg.text = nsprintf( @"%d", tt);
}

// Debug on/off
//----------------------------
- (void) swiDbg:(id) sender
{
    self.debug_mode = [self.swiDbg isOn];
}

- (void) btnGo: (id) sender { if (self.debug_mode) [self debugFlow:false]; }

// Camera button press
//------------------------------
- (void) btnCam:(id)sender
{
    g_app.saveDiscardVC.photo = [self.cameraView.image copy];
    g_app.saveDiscardVC.sgf = [g_app.mainVC.cppInterface get_sgf];
    [g_app.navVC pushViewController:g_app.saveDiscardVC animated:YES];
}

// Debugging helper, shows individual processing stages
//------------------------------------------------------
- (void) debugFlow:(bool)reset
{
    if (reset) _debugstate = 0;
    UIImage *img;
    while(1) {
        switch (_debugstate) {
            case 0:
                _debugstate++;
                self.frame_grabber_on = NO;
                [self.frameExtractor suspend];
                img = [self.cppInterface f00_blobs];
                [self.cameraView setImage:img];
                break;
            case 1:
                img = [self.cppInterface f01_vert_lines];
                if (!img) { _debugstate=2; continue; }
                [self.cameraView setImage:img];
                break;
            case 2:
                img = [self.cppInterface f02_horiz_lines];
                if (!img) { _debugstate=3; continue; }
                [self.cameraView setImage:img];
                break;
            case 3:
                _debugstate++;
                img = [self.cppInterface f03_corners];
                [self.cameraView setImage:img];
                break;
            case 4:
                _debugstate++;
                img = [self.cppInterface f04_zoom_in];
                [self.cameraView setImage:img];
                break;
            case 5:
                _debugstate++;
                img = [self.cppInterface f05_dark_places];
                [self.cameraView setImage:img];
                break;
            case 6:
                _debugstate++;
                img = [self.cppInterface f06_mask_dark];
                [self.cameraView setImage:img];
                break;
            case 7:
                _debugstate++;
                img = [self.cppInterface f07_white_holes];
                [self.cameraView setImage:img];
                break;
            case 8:
                _debugstate++; continue; // skip;
                //                    img = [self.cppInterface f08_features];
                //                    if (!img) { _sliderstate=9; continue; }
                //                    [self.cameraView setImage:img];
                break;
            case 9:
                _debugstate++;
                img = [self.cppInterface f09_classify];
                [self.cameraView setImage:img];
                break;
            default:
                _debugstate=0;
                continue;
                //self.frame_grabber_on = YES;
                //[self.frameExtractor resume];
                //g_app.mainVC.lbDbg.text = @"";
        } // switch
        break;
    } // while(1)
} // debugFlow()

#pragma mark - FrameExtractorDelegate protocol
//-----------------------------------------------
- (void)captured:(UIImage *)image
{
    if ([g_app.menuVC debugMode]) {
        self.frame_grabber_on = NO;
        [self.frameExtractor suspend];
        return;
    }
    if (self.frame_grabber_on) {
        if (self.debug_mode) {
            [self.cameraView setImage:image];
            _img = image;
            [_cppInterface qImg:_img];
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


@end
