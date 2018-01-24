//
//  SaveDiscardVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-19.
//  Copyright © 2018 AHN. All rights reserved.
//

#import "SaveDiscardVC.h"
#import "Common.h"
#import "Globals.h"
#import "ImagesVC.h"

@interface SaveDiscardVC ()
@property UIImage *sgfImg;
@property UIImageView *sgfView;
@property UIImageView *photoView;
@property UIButton *btnDiscard;
@property UIButton *btnUse;
@end

@implementation SaveDiscardVC
//-----------------------------
- (id)init
{
    self = [super init];
    if (self) {
        self.view = [UIView new];
        UIView *v = self.view;
        v.autoresizesSubviews = NO;
        v.opaque = YES;
        v.backgroundColor = BGCOLOR;
        
        // Image View for photo
        _photoView = [UIImageView new];
        _photoView.contentMode = UIViewContentModeScaleAspectFit;
        [v addSubview:_photoView];
        
        // Image View for sgf
        _sgfView = [UIImageView new];
        _sgfView.contentMode = UIViewContentModeScaleAspectFit;
        [v addSubview:_sgfView];
        
        // Buttons
        //=========
        // Use
        _btnUse = [UIButton new];
        [_btnUse setTitle:@"Use" forState:UIControlStateNormal];
        [_btnUse.titleLabel setFont:[UIFont fontWithName:@"HelveticaNeue" size:30.0]];
        [_btnUse sizeToFit];
        [_btnUse addTarget:self action:@selector(btnUse:) forControlEvents: UIControlEventTouchUpInside];
        [v addSubview:_btnUse];
        // Discard
        _btnDiscard = [UIButton new];
        [_btnDiscard setTitle:@"Discard" forState:UIControlStateNormal];
        [_btnDiscard setTitleColor:RED forState:UIControlStateNormal];
        [_btnDiscard.titleLabel setFont:[UIFont fontWithName:@"HelveticaNeue" size:30.0]];
        [_btnDiscard sizeToFit];
        [_btnDiscard addTarget:self action:@selector(btnDiscard:) forControlEvents: UIControlEventTouchUpInside];
        [v addSubview:_btnDiscard];
    }
    return self;
} // init()

//----------------------
- (void)viewDidLoad
{
    [super viewDidLoad];
}

 //----------------------------------------
 - (void) viewDidAppear: (BOOL) animated
 {
     [super viewDidAppear: animated];
     [self doLayout];
 }

//----------------------------------
- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
}

// Button Callbacks
//======================

//---------------------------
- (void) btnUse:(id)sender
{
    // Make filename from date
    NSDictionary *tstamp = dateAsDict();
    NSString *fname = nsprintf( @"%4d%02d%02d-%02d%02d%02d.png",
                               [tstamp[@"year"] intValue],
                               [tstamp[@"month"] intValue],
                               [tstamp[@"day"] intValue],
                               [tstamp[@"hour"] intValue],
                               [tstamp[@"minute"] intValue],
                               [tstamp[@"second"] intValue]);
    fname = nsprintf( @"%@/%@", @SAVED_FOLDER, fname);
    fname = getFullPath( fname);
    // Save png
    [UIImagePNGRepresentation(_photo) writeToFile:fname atomically:YES];
    // Save sgf
    fname = changeExtension( fname, @".sgf");
    NSError *error;
    [_sgf writeToFile:fname
           atomically:YES encoding:NSUTF8StringEncoding error:&error];
    // TODO: This should really got to ImagesVC
    [g_app.navVC popViewControllerAnimated:NO];
    [g_app.navVC pushViewController:g_app.imagesVC animated:YES];
} // btnUse()

//------------------------------
- (void) btnDiscard:(id)sender
{
    [g_app.navVC popViewControllerAnimated:YES];
} // btnDiscard()

// Layout
//==========

// Put UI elements into the right place
//---------------------------------------
- (void) doLayout
{
    float W = SCREEN_WIDTH;
    float topmarg = g_app.navVC.navigationBar.frame.size.height;
    float lmarg = W/40;
    float rmarg = W/40;
    float sep = W/40;
    float imgWidth = (W  - lmarg - rmarg) / 2 - sep;
    
    // Photo view
    _photoView.frame = CGRectMake( lmarg, topmarg + 40, imgWidth , imgWidth);
    _photoView.hidden = NO;
    if (_photo) {
        [_photoView setImage:_photo];
    }
    // Sgf View
    _sgfView.hidden = NO;
    _sgfView.frame = CGRectMake( lmarg + imgWidth + sep, topmarg + 40, imgWidth , imgWidth);
    if (_sgf) {
        _sgfImg = [CppInterface sgf2img:_sgf];
        [_sgfView setImage:_sgfImg];
    }
    // Buttons
    float btnWidth, btnHeight;
    int y = topmarg + 40 + imgWidth + 100;;
    
    _btnUse.hidden = NO;
    [_btnUse setTitleColor:self.view.tintColor forState:UIControlStateNormal];
    btnWidth = _btnUse.frame.size.width;
    btnHeight = _btnUse.frame.size.height;
    _btnUse.frame = CGRectMake( W/2 - btnWidth/2, y, btnWidth, btnHeight);

    y += btnHeight * 1.5;
    _btnDiscard.hidden = NO;
    [_btnDiscard setTitleColor:RED forState:UIControlStateNormal];
    btnWidth = _btnDiscard.frame.size.width;
    btnHeight = _btnDiscard.frame.size.height;
    _btnDiscard.frame = CGRectMake( W/2 - btnWidth/2, y, btnWidth, btnHeight);
} // doLayout()

@end




































