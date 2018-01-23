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

@interface SaveDiscardVC ()
@property UIImage *sgfImg;
@property UIImageView *sgfView;
@property UIImageView *photoView;
@property UIButton *btnDiscard;
@property UIButton *btnSave;
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
        // Save
        _btnSave = [UIButton new];
        [_btnSave setTitle:@"Save" forState:UIControlStateNormal];
        [_btnSave.titleLabel setFont:[UIFont fontWithName:@"HelveticaNeue" size:30.0]];
        [_btnSave sizeToFit];
        [v addSubview:_btnSave];
        // Discard
        _btnDiscard = [UIButton new];
        [_btnDiscard setTitle:@"Discard" forState:UIControlStateNormal];
        [_btnDiscard setTitleColor:RED forState:UIControlStateNormal];
        [_btnDiscard.titleLabel setFont:[UIFont fontWithName:@"HelveticaNeue" size:30.0]];
        [_btnDiscard sizeToFit];
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

#pragma mark Layout

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
    _btnDiscard.hidden = NO;
    [_btnDiscard setTitleColor:RED forState:UIControlStateNormal];
    btnWidth = _btnDiscard.frame.size.width;
    btnHeight = _btnDiscard.frame.size.height;
    int y = topmarg + 40 + imgWidth + 100;
    _btnDiscard.frame = CGRectMake( W/2 - btnWidth/2, y, btnWidth, btnHeight);
} // doLayout()

@end




































