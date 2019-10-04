import React from 'react';
import * as PropTypes from 'prop-types';
import Paper from '@material-ui/core/Paper';
import { FilePond } from 'react-filepond';
import Grid from '@material-ui/core/Grid';
import CardHeader from '@material-ui/core/CardHeader';
import withStyles from '@material-ui/core/styles/withStyles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/es/CardContent/CardContent';
import TextField from '@material-ui/core/TextField';
import className from 'classnames';
import Popper from '@material-ui/core/Popper';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import Grow from '@material-ui/core/Grow';
import RequiredField from '../../../common/RequiredField';
import Form from '../../../common/Form';
import Button from '../../../common/Button';
import CollapseGrow from '../../../common/CollapseGrow';
import { VideoList } from './NewSearchPaper';

const styles = theme => ({
  root: {
    backgroundColor: 'transparent',
    width: '90%',
    margin: 'auto',
    padding: theme.spacing.unit,
  },
  button: {
    width: '100%',
  },
  filter: {
    minHeight: 200,
    minWidth: 360,
  },
  card: {
    marginBottom: 2 * theme.spacing.unit,
    position: 'relative',
    '&:last-child': {
      marginBottom: 0,
    },
  },
  imgCard: {
    width: 210,
    height: 118,
  },
  finish: {
    position: 'absolute',
    right: 0,
  },
  menu: {
    width: 230,
    padding: theme.spacing.unit * 2,
    outline: 'none',
    borderRadius: theme.shape.buttonBorderRadius,
  },
  videoListRoot: {
    width: '100%',
    padding: theme.spacing.unit,
    maxHeight: 800,
    overflowY: 'scroll',
  },
});

class SearchPaper extends Form {
  state = {
    anchorEl: null,
    textContent: '',
    textDisplay: true,
    imgContent: [],
    imageDisplay: true,
    faceContent: [],
    faceDisplay: true,
    video: true,
    videoContent: [],
  };

  child = {
    text: React.createRef(),
    image: React.createRef(),
  };

  componentWillMount() {
    const { handleGetVideoList } = this.props;
    handleGetVideoList();
  }

  handleClose = e => {
    const { anchorEl } = this.state;

    if (anchorEl.contains(e.target)) {
      return;
    }

    this.setState({ anchorEl: null });
  };

  handleContentChange = name => (e, callback) =>
    this.setState({ [name]: e.target.value }, callback);

  handleVideoContentChange = id => e => {
    const { checked } = e.target;
    this.setState(state => {
      const { videoContent: _v } = state;
      const videoContent = _v.slice();
      if (checked) videoContent.push(id);
      else {
        const index = videoContent.indexOf(id);
        if (index > -1) {
          videoContent.splice(index, 1);
        }
      }
      return { videoContent };
    });
  };

  handleSubmit = this.handleSubmitWrapper(() => {
    const { textContent, imgContent, videoContent } = this.state;
    this.props.handleSearch({
      text: textContent,
      img: imgContent.length ? imgContent[0] : null,
      target: videoContent,
    });
  });

  render() {
    const { classes, theme, videos } = this.props;
    const { anchorEl, textContent, imgContent, videoContent } = this.state;
    const open = Boolean(anchorEl);

    return (
      <Paper className={classes.root} elevation={0}>
        <Grid className={classes.card} spacing={8} container>
          <Grid xs md={12} lg item>
            <CollapseGrow in>
              <Card className={className(classes.card, classes.filter)}>
                <CardHeader title='Product Image' aria-label='product image' />
                <CardContent>
                  <RequiredField
                    value={imgContent.length}
                    initValue={0}
                    ref={this.child.image}
                    bindingChange='onupdatefiles'
                  >
                    <FilePond
                      maxFiles={1}
                      onupdatefiles={(items, callback) => {
                        this.setState(
                          {
                            imgContent: items.map(item => item.file),
                          },
                          callback,
                        );
                      }}
                      acceptedFileTypes={['image/*']}
                    />
                  </RequiredField>
                </CardContent>
              </Card>
            </CollapseGrow>
          </Grid>

          <Grid xs md={12} lg item>
            <CollapseGrow in>
              <Card className={className(classes.card, classes.filter)}>
                <CardHeader title='Face Image' aria-label='face image' />
                <CardContent>
                  <FilePond
                    maxFiles={1}
                    onupdatefiles={items => {
                      this.setState({
                        sceneContent: items.map(item => item.file),
                      });
                    }}
                    acceptedFileTypes={['image/*']}
                  />
                </CardContent>
              </Card>
            </CollapseGrow>
          </Grid>

          <Grid xs md={12} lg item>
            <CollapseGrow in>
              <Card className={className(classes.card, classes.filter)}>
                <CardHeader
                  title='Product Description'
                  aria-label='product description'
                />
                <CardContent>
                  <RequiredField initValue='' ref={this.child.text}>
                    <TextField
                      id='searchPaper-text-input'
                      label='Text'
                      placeholder='Any text here...'
                      value={textContent}
                      onChange={this.handleContentChange('textContent')}
                      multiline
                      fullWidth
                      margin='normal'
                    />
                  </RequiredField>
                </CardContent>
              </Card>
            </CollapseGrow>
          </Grid>
        </Grid>

        <Grid className={classes.card} justify='flex-end' spacing={8} container>
          <Grid item>
            <Button
              aria-owns={open ? 'fade-menu' : undefined}
              aria-haspopup='true'
              onClick={e => {
                const { currentTarget } = e;
                this.setState(state => {
                  if (!state.anchorEl)
                    return {
                      anchorEl: currentTarget,
                    };
                  return { anchorEl: null };
                });
              }}
            >
              {videoContent.length ? '● ' : ''}
              Video List ▼
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant='contained'
              color='primary'
              onClick={this.handleSubmit}
            >
              Search
            </Button>
          </Grid>
        </Grid>

        <Popper
          open={open}
          anchorEl={anchorEl}
          onClose={this.handleClose}
          style={{ zIndex: theme.zIndex.drawer + 1 }}
          transition
        >
          {({ TransitionProps }) => (
            <Grow
              {...TransitionProps}
              id='menu-list-grow'
              style={{
                transformOrigin: 'center top',
              }}
            >
              <Paper className={classes.menu}>
                <ClickAwayListener onClickAway={this.handleClose}>
                  <VideoList
                    classes={{ root: classes.videoListRoot }}
                    updateSelection={this.handleVideoContentChange}
                    videos={videos}
                    videoContent={videoContent}
                  />
                </ClickAwayListener>
              </Paper>
            </Grow>
          )}
        </Popper>
      </Paper>
    );
  }
}

SearchPaper.propTypes = {
  classes: PropTypes.object.isRequired,
  videos: PropTypes.array.isRequired,
  handleSearch: PropTypes.func.isRequired,
  handleGetVideoList: PropTypes.func.isRequired,
};

export default withStyles(styles, { withTheme: true })(SearchPaper);
