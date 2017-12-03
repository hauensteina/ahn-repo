//
//  MainVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//

#import "MainVC.h"

#import "Globals.h"
#import "CppInterface.h"

#define DDEBUG

//==========================
@interface MainVC ()
@property FrameExtractor *frameExtractor;
@property CppInterface *grabFuncs;
@property UIImageView *cameraView;
// Data
@property UIImage *img; // The current image
// Buttons etc
@property UIButton *btnGo;
@property UISlider *sldCannyLow;
@property UISlider *sldCannyHi;
@property UISwitch *swiDbg;

// State
@property BOOL frame_grabber_on; // Set to NO to stop the frame grabber
@property BOOL debug_mode;

@end

//=========================
@implementation MainVC

//----------------------
- (void)viewDidLoad
{
    [super viewDidLoad];
    self.frameExtractor = [FrameExtractor new];
    self.grabFuncs = [CppInterface new];
    self.frameExtractor.delegate = self;
    self.frame_grabber_on = YES;
    
    //self.sldCannyLow.value = self.grabFuncs.canny_low;
    //self.sldCannyHi.value  = self.grabFuncs.canny_hi;
    //NSString *tstr = [GrabFuncs opencvVersion];
    //NSLog(tstr);
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
    [swi setOn:NO];
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
    
    // Canny low slider
    UISlider *s = [UISlider new];
    self.sldCannyLow = s;
    s.minimumValue = 0;
    s.maximumValue = 0.5;
    //s.maximumValue = 10;
    [s addTarget:self action:@selector(sldCannyLow:) forControlEvents:UIControlEventValueChanged];
    s.backgroundColor = RGB (0xf0f0f0);
    [v addSubview:s];
    self.sldCannyLow.hidden = true;

    // Canny high slider
    s = [UISlider new];
    self.sldCannyHi = s;
    s.minimumValue = 0;
    s.maximumValue = 255;
    [s addTarget:self action:@selector(sldCannyHi:) forControlEvents:UIControlEventValueChanged];
    s.backgroundColor = RGB (0xf0f0f0);
    [v addSubview:s];
    self.sldCannyHi.hidden = true;
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
    // Canny hi slider
    y -= delta_y;
    self.sldCannyHi.frame = CGRectMake(lmarg, y, W - lmarg - rmarg, mh);
    // Canny low slider
    y -= delta_y;
    self.sldCannyLow.frame = CGRectMake(lmarg, y, W - lmarg - rmarg, mh);

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
    b.titleLabel.font = g_fntBtn;
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

// Slider for low canny threshold
//-----------------------------------
- (void) sldCannyLow:(id) sender
{
    float tt = [self.sldCannyLow value];
    self.grabFuncs.sld_low = tt;
    //self.grabFuncs.thresh = tt;
    self.lbDbg.text = [NSString stringWithFormat:@"%.2f %d", tt, self.grabFuncs.canny_hi];
}

// Slider for hi canny threshold
//-----------------------------------
- (void) sldCannyHi:(id) sender
{
    int tt = [self.sldCannyHi value];
    self.grabFuncs.canny_hi = tt;
    self.lbDbg.text = [NSString stringWithFormat:@"%.2f %d", self.grabFuncs.sld_low, tt];
}

// Debug on/off
//-----------------------------------
- (void) swiDbg:(id) sender
{
    BOOL tt = [self.swiDbg isOn];
    self.debug_mode = tt;
}

// Debugging helper, shows individual processing stages
//------------------------------------------------------
- (void) btnGo: (id) sender
{
    if (self.debug_mode) {
        static int state = 0;
        UIImage *img;
        switch (state) {
            case 0:
                state++;
                //state=100;
                self.frame_grabber_on = NO;
                [self.frameExtractor suspend];
                img = [self.grabFuncs f00_blobs:self.img];
                [self.cameraView setImage:img];
                break;
            case 1:
                state++;
                img = [self.grabFuncs f01_straight];
                [self.cameraView setImage:img];
                break;
            case 2:
                state++;
                img = [self.grabFuncs f02_horiz_lines];
                [self.cameraView setImage:img];
                break;
            case 3:
                state++;
                img = [self.grabFuncs f03_vert_lines];
                [self.cameraView setImage:img];
                break;
            case 4:
                state++;
                img = [self.grabFuncs f04_vert_lines_2];
                [self.cameraView setImage:img];
                break;
            case 5:
                state++;
                img = [self.grabFuncs f05_vert_params];
                [self.cameraView setImage:img];
                break;
            case 6:
                state++;
                img = [self.grabFuncs f06_corners];
                [self.cameraView setImage:img];
                break;
            case 7:
                state++;
                img = [self.grabFuncs f07_zoom_in];
                [self.cameraView setImage:img];
                break;
//            case 8:
//                state++;
//                img = [self.grabFuncs f08_show_vert_lines];
//                [self.cameraView setImage:img];
//                break;
//            case 9:
//                state++;
//                img = [self.grabFuncs f09_clean_horiz_lines];
//                [self.cameraView setImage:img];
//                break;
//            case 10:
//                state++;
//                img = [self.grabFuncs f10_clean_vert_lines];
//                [self.cameraView setImage:img];
//                break;
//            case 11:
//                state++;
//                img = [self.grabFuncs f11_classify];
//                [self.cameraView setImage:img];
//                break;
            default:
                state=0;
                self.frame_grabber_on = YES;
                [self.frameExtractor resume];
        } // switch
    }
} // btnGo()


#pragma mark - FrameExtractorDelegate protocol
//---------------------------------
- (void)captured:(UIImage *)image
{
    //self.cameraView.hidden = NO;
    //static int i = 0;
    if (self.frame_grabber_on) {
        //i++;
        //PLOG("frame:%d\n",i);
        if (self.debug_mode) {
            [self.cameraView setImage:image];
            self.img = image;
        }
        else {
            self.frame_grabber_on = NO;
            [self.frameExtractor suspend];
            UIImage *processedImg = [self.grabFuncs findBoard:image];
            self.img = processedImg;
            [self.cameraView setImage:self.img];
            self.frame_grabber_on = YES;
            [self.frameExtractor resume];
        }
    }
}


@end
