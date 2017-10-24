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
// Data
@property UIImage *img; // The current image
// Buttons
@property UIButton *btnGo;
// State
@property BOOL frame_grabber_on; // Set to NO to stop the frame grabber

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
    self.frame_grabber_on = YES;
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
    v.backgroundColor = RGB(0xF3DCA5);
    
    // Camera View
    self.cameraView = [UIImageView new];
    [v addSubview:self.cameraView];
    
    // Buttons
    self.btnGo = [self addButtonWithTitle:@"Go" callback:@selector(btnGo:)];
}

//----------------------------------------------------------------------
- (void) viewWillAppear:(BOOL) animated
{
    [super viewWillAppear: animated];
    [self doLayout];
    
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

    self.btnGo.frame = CGRectMake (lmarg, y, W - lmarg - rmarg, mh);
    self.btnGo.titleLabel.font = [UIFont fontWithName:@"HelveticaNeue" size: 40];
    [self.btnGo setTitleColor:DARKRED forState:UIControlStateNormal];
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

#pragma mark - Button callbacks

// Debugging helper, shows individual processing stages
//------------------------------------------------------
- (void) btnGo: (id) sender
{
#ifdef DDEBUG
    static int state = 0;
    UIImage *img;
    switch (state) {
        case 0:
            state++;
            self.frame_grabber_on = NO;
            [self.frameExtractor suspend];
            img = [self.grabFuncs f00_contours:self.img];
            [self.cameraView setImage:img];
            break;
        case 1:
            state++;
            img = [self.grabFuncs f01_filtered_contours];
            [self.cameraView setImage:img];
            break;
        case 2:
            state++;
            img = [self.grabFuncs f02_inside_contours];
            [self.cameraView setImage:img];
            break;
        case 3:
            state++;
            img = [self.grabFuncs f03_find_board];
            [self.cameraView setImage:img];
            break;
        default:
            state=0;
            self.frame_grabber_on = YES;
            [self.frameExtractor resume];
    } // switch
#endif
} // btnGo()


#pragma mark - FrameExtractorDelegate protocol
//---------------------------------
- (void)captured:(UIImage *)image
{
    //self.cameraView.hidden = NO;
    if (self.frame_grabber_on) {
#ifdef DDEBUG
        [self.cameraView setImage:image];
        self.img = image;
#else
        self.frame_grabber_on = NO;
        UIImage *processedImg = [self.grabFuncs findBoard:image];
        self.img = processedImg;
        [self.cameraView setImage:self.img];
        self.frame_grabber_on = YES;
#endif
    }
}


@end
