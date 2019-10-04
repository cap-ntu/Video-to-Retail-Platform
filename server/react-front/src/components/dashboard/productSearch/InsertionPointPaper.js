/* react/no-array-index-key,react/jsx-wrap-multilines,react/jsx-indent,react/jsx-indent-props */
import React from 'react';
import * as PropTypes from 'prop-types';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import withStyles from '@material-ui/core/styles/withStyles';
import Card from '@material-ui/core/Card';
import CardMedia from '@material-ui/core/CardMedia';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/es/Divider/Divider';
import TextField from '@material-ui/core/TextField';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { isEmpty, secondToHMS } from '../../../utils/utils';
import Button from '../../common/Button';

const styles = theme => ({
  root: {
    backgroundColor: 'transparent',
    margin: 'auto',
    padding: theme.spacing.unit,
  },
  card: {
    display: 'inline-block',
  },
  cardPaper: {
    background: 'transparent',
  },
  imgCard: {
    width: 210,
    height: 118,
  },
  videoGrid: {
    width: 210,
    padding: 0.5 * theme.spacing.unit,
  },
  divider: {
    margin: [['5px', '0px']],
    '&:last-child': {
      display: 'none',
    },
  },
});

class InsertionPointPaper extends React.PureComponent {
  state = { bidPrice: {} };

  componentWillMount() {
    const { websocketOnMessage } = this.props;
    const wsPath = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${
      window.location.host
    }/bidding/stream/`;
    const socket = new ReconnectingWebSocket(wsPath);
    socket.onmessage = websocketOnMessage;
    this.socket = socket;
  }

  componentWillUnmount() {
    this.socket.close();
  }

  handleUpdatePrice = id => event => {
    const { value } = event.target;
    this.setState(state => ({
      ...state.bidPrice,
      bidPrice: { [id.toString()]: value },
    }));
  };

  handleBid = (id, price) => {
    this.socket.send(JSON.stringify({ item_id: id, price }));
  };

  render() {
    const { classes, videos, handleInsert } = this.props;
    const { bidPrice } = this.state;

    return (
      <Paper className={classes.root} elevation={0}>
        <Typography variant='h6' gutterBottom>
          Search Result
        </Typography>
        {isEmpty(videos) ? (
          <Typography variant='subtitle2' color='textSecondary'>
            No result yet, tune your filter condition and try again.
          </Typography>
        ) : null}
        <div>
          {Object.keys(videos).map(key => (
            <React.Fragment key={key}>
              <Typography component='div' variant='subtitle1'>
                {key}
              </Typography>
              <Grid spacing={8} container>
                {videos[key].map(clip => (
                  <Grid key={clip.id} item>
                    <Grid className={classes.videoGrid} item>
                      <Card
                        className={classes.card}
                        elevation={0}
                        title='preview'
                      >
                        <CardMedia
                          className={classes.imgCard}
                          component='img'
                          src={clip.cover}
                          alt={clip.name}
                        />
                      </Card>
                    </Grid>
                    <Grid className={classes.videoGrid} item>
                      <Typography gutterBottom>{clip.id}</Typography>
                      <Grid direction='row' spacing={8} container>
                        <Grid item xs>
                          <Typography>
                            {`Time: ${secondToHMS(clip.start)} - ${secondToHMS(
                              clip.end,
                            )}`}
                          </Typography>
                        </Grid>
                        <Grid item>
                          <Button
                            variant='contained'
                            color='primary'
                            size='small'
                            onClick={() => handleInsert(clip.id)}
                          >
                            Insert
                          </Button>
                        </Grid>
                      </Grid>
                      <Typography
                        variant='subtitle1'
                        color='error'
                        gutterBottom
                      >
                        {`$${clip.price}`}
                      </Typography>
                      <TextField
                        id={`${clip.id}`}
                        label='Your bidding price'
                        value={bidPrice[`${clip.id}`] || clip.price}
                        onChange={this.handleUpdatePrice(clip.id)}
                        type='number'
                        className={classes.textField}
                        InputLabelProps={{
                          shrink: true,
                        }}
                        margin='normal'
                        style={{ width: '100%' }}
                      />
                      <Button
                        variant='contained'
                        color='primary'
                        size='small'
                        onClick={() =>
                          this.handleBid(clip.id, bidPrice[clip.id])
                        }
                        style={{ width: '100%' }}
                      >
                        Bid Now
                      </Button>
                    </Grid>
                  </Grid>
                ))}
              </Grid>
              <Divider className={classes.divider} />
            </React.Fragment>
          ))}
        </div>
      </Paper>
    );
  }
}

InsertionPointPaper.defaultProps = {
  videos: {},
};

InsertionPointPaper.propTypes = {
  classes: PropTypes.object.isRequired,
  videos: PropTypes.shape({
    [PropTypes.string]: PropTypes.arrayOf(
      PropTypes.shape({
        id: PropTypes.string.isRequired,
        name: PropTypes.string.isRequired,
        cover: PropTypes.string,
        start: PropTypes.number.isRequired,
        end: PropTypes.number.isRequired,
      }).isRequired,
    ).isRequired,
  }),
  handleInsert: PropTypes.func.isRequired,
  websocketOnMessage: PropTypes.func.isRequired,
};

export default withStyles(styles)(InsertionPointPaper);
